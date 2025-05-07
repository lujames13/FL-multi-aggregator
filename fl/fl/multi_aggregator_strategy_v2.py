"""fl: A Flower / PyTorch app with multiple virtual aggregators."""

import copy
import json
import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import (
    Context,
    EvaluateRes,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, Strategy
from .multi_aggregator_strategy import MultiAggregatorStrategy

logger = logging.getLogger(__name__)


class MultiAggregatorStrategy_v2(MultiAggregatorStrategy):
    """增強版多聚合器策略，支持延遲檢測、回滾機制，和惡意Aggregator剔除。"""
    
    def __init__(
        self,
        num_aggregators: int = 3,
        malicious_aggregator_ids: List[int] = None,
        enable_challenges: bool = True,
        detection_delay: int = 2,  # 檢測延遲輪數
        **kwargs,
    ):
        """初始化增強版多聚合器策略。"""
        super().__init__(
            num_aggregators=num_aggregators,
            malicious_aggregator_ids=malicious_aggregator_ids,
            enable_challenges=enable_challenges,
            **kwargs,
        )
        
        # 延遲檢測相關屬性
        self.detection_delay = detection_delay
        self.pending_detections = {}  # 格式: {檢測輪: (惡意輪, 惡意聚合器ID, 參數)}
        self.detected_attacks = set()  # 已檢測到的攻擊輪
        self.excluded_aggregators = set()  # 被剔除的聚合器ID
        self.safe_parameters_history = {}  # 安全參數歷史 {輪: 參數}
        self.original_rounds_map = {}  # 映射 {當前輪: 原始輪} (用於回滾後)
        self.total_rounds_with_attack = 0  # 攻擊情況下的總輪數
        self.total_rollbacks = 0  # 總回滾次數
        self.is_recovery_mode = False  # 是否處於恢復模式
        self.recovery_from_round = None  # 從哪一輪恢復
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """聚合來自客戶端的模型更新。"""
        self.round = server_round
        
        # 跟踪原始輪映射 (用於研究指標)
        if server_round not in self.original_rounds_map:
            if self.is_recovery_mode:
                # 恢復模式的映射邏輯
                previous_round = server_round - 1
                if previous_round in self.original_rounds_map:
                    self.original_rounds_map[server_round] = self.original_rounds_map[previous_round]
                else:
                    self.original_rounds_map[server_round] = self.recovery_from_round
            else:
                # 正常模式，此輪就是原始輪
                self.original_rounds_map[server_round] = server_round
        
        # 步驟 1: 檢查待處理的檢測
        if server_round in self.pending_detections:
            malicious_round, malicious_agg_id, honest_parameters = self.pending_detections[server_round]
            logger.info(f"檢測輪 {server_round}: 檢查輪 {malicious_round} 的惡意聚合器 {malicious_agg_id}")
            
            # 模擬挑戰驗證 (在實際應用中，這將是基於證明的驗證)
            challenge_success = True  # 在此模擬中，我們假設所有延遲檢測都成功
            
            if challenge_success:
                logger.warning(f"檢測到攻擊! 在輪 {malicious_round} 由聚合器 {malicious_agg_id}，將在下一輪回滾並剔除此聚合器")
                self.detected_attacks.add(malicious_round)
                self.excluded_aggregators.add(malicious_agg_id)
                self.total_rollbacks += 1
                
                # 設置恢復模式，下一輪從此處恢復
                self.is_recovery_mode = True
                self.recovery_from_round = malicious_round - 1
                
                # 記錄檢測和剔除事件
                challenge_metrics = {
                    "round": int(server_round),
                    "detected_malicious_round": int(malicious_round),
                    "malicious_aggregator_id": int(malicious_agg_id),
                    "challenged": True,
                    "challenge_success": True,
                    "rollback_scheduled": True,
                    "aggregator_excluded": True
                }
                self.metrics_history.append(challenge_metrics)
                
                # 如果所有惡意聚合器都被剔除，更新原始列表以供指標使用
                remaining_malicious = [agg_id for agg_id in self.malicious_aggregator_ids 
                                      if agg_id not in self.excluded_aggregators]
                logger.info(f"剩餘惡意聚合器: {remaining_malicious}")
                logger.info(f"已剔除聚合器: {self.excluded_aggregators}")
            
            # 清除此檢測
            del self.pending_detections[server_round]
        
        # 步驟 2: 確定聚合器和處理恢復模式
        if self.is_recovery_mode:
            logger.info(f"輪 {server_round}: 恢復模式激活，從輪 {self.recovery_from_round} 恢復")
            
            # 獲取安全參數作為起點
            if self.recovery_from_round in self.safe_parameters_history:
                safe_parameters = self.safe_parameters_history[self.recovery_from_round]
                
                # 在恢復模式下，重新選擇有效的聚合器（排除被剔除的）
                while True:
                    self.current_aggregator_id = (server_round - 1) % self.num_aggregators
                    if self.current_aggregator_id not in self.excluded_aggregators:
                        break
                    # 如果當前輪次對應的聚合器已被剔除，模擬輪次+1繼續尋找
                    server_round += 1
                
                # 檢查選擇的聚合器是否惡意（理論上應該都被剔除了，但以防萬一）
                is_malicious = self.current_aggregator_id in self.malicious_aggregator_ids
                
                logger.info(f"恢復聚合: 使用聚合器 {self.current_aggregator_id}" +
                          (" (惡意，但尚未被剔除)" if is_malicious else ""))
                
                # 使用父類進行誠實聚合，但從安全參數開始
                aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
                
                # 標記此為恢復輪
                metrics["recovery_round"] = True
                metrics["recovered_from_round"] = self.recovery_from_round
                metrics["aggregator_id"] = self.current_aggregator_id
                metrics["excluded_aggregators"] = list(self.excluded_aggregators)
                
                # 恢復完成，退出恢復模式
                self.is_recovery_mode = False
                
                # 將此輪的參數存儲為安全參數
                self._store_safe_parameters(server_round, aggregated_parameters)
                
                return aggregated_parameters, metrics
            else:
                logger.error(f"無法找到輪 {self.recovery_from_round} 的安全參數，退出恢復模式")
                self.is_recovery_mode = False
        
        # 步驟 3: 選擇未被剔除的聚合器
        # 找到一個未被剔除的聚合器
        original_round = server_round
        while True:
            self.current_aggregator_id = (server_round - 1) % self.num_aggregators
            if self.current_aggregator_id not in self.excluded_aggregators:
                break
            # 如果當前輪次對應的聚合器已被剔除，模擬輪次+1繼續尋找
            server_round += 1
        
        # 如果我們不得不調整輪次來找到有效聚合器，記錄這一點
        if server_round != original_round:
            logger.info(f"原輪次 {original_round} 對應的聚合器已被剔除，跳至虛擬輪次 {server_round} 使用聚合器 {self.current_aggregator_id}")
        
        is_malicious = self.current_aggregator_id in self.malicious_aggregator_ids
        
        logger.info(f"輪 {original_round} (虛擬輪次 {server_round}): 使用聚合器 {self.current_aggregator_id}" +
                   (" (惡意)" if is_malicious else ""))
        
        # 首先計算誠實聚合 (用於比較和挑戰)
        honest_parameters, honest_metrics = super().aggregate_fit(original_round, results, failures)
        if honest_parameters is None:
            return None, {}
        
        # 存儲此輪的安全參數 (用於未來可能的恢復)
        self._store_safe_parameters(original_round, honest_parameters)
        
        # 如果這是惡意聚合器，操縱結果
        if is_malicious:
            honest_ndarrays = parameters_to_ndarrays(honest_parameters)
            malicious_ndarrays = self._create_malicious_aggregation(honest_ndarrays)
            aggregated_parameters = ndarrays_to_parameters(malicious_ndarrays)
            
            # 添加標籤以識別惡意行為
            metrics = {
                **honest_metrics,
                "aggregator_id": self.current_aggregator_id,
                "malicious": True,
                "excluded_aggregators": list(self.excluded_aggregators)
            }
            
            # 安排未來輪次的延遲檢測
            detection_round = original_round + self.detection_delay
            self.pending_detections[detection_round] = (
                original_round, 
                self.current_aggregator_id, 
                honest_parameters
            )
            logger.info(f"輪 {original_round}: 惡意聚合器 {self.current_aggregator_id}，安排在輪 {detection_round} 進行檢測")
            
            # 增加攻擊情況下的總輪數計數
            self.total_rounds_with_attack += 1
        else:
            aggregated_parameters = honest_parameters
            metrics = {
                **honest_metrics,
                "aggregator_id": self.current_aggregator_id,
                "malicious": False,
                "excluded_aggregators": list(self.excluded_aggregators)
            }
        
        return aggregated_parameters, metrics
    
    def _store_safe_parameters(self, server_round: int, parameters: Parameters) -> None:
        """存儲安全參數用於未來可能的恢復。"""
        self.safe_parameters_history[server_round] = parameters
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """聚合來自客戶端的評估結果。"""
        # 確保使用未被剔除的聚合器
        original_round = server_round
        while True:
            current_agg_id = (server_round - 1) % self.num_aggregators
            if current_agg_id not in self.excluded_aggregators:
                self.current_aggregator_id = current_agg_id
                break
            server_round += 1
        
        # 執行父類的評估聚合
        aggregated_loss, metrics = super().aggregate_evaluate(original_round, results, failures)
        
        # 添加聚合器信息到指標
        metrics["aggregator_id"] = self.current_aggregator_id
        metrics["malicious"] = self.current_aggregator_id in self.malicious_aggregator_ids
        metrics["challenged"] = original_round in self.challenged_rounds
        metrics["excluded_aggregators"] = list(self.excluded_aggregators)
        metrics["effective_aggregators"] = self.num_aggregators - len(self.excluded_aggregators)
        
        # 每5輪或首輪生成研究指標摘要
        if original_round % 5 == 0 or original_round == 1:
            self._log_research_metrics()
        
        return aggregated_loss, metrics
    
    def _log_research_metrics(self) -> None:
        """記錄與研究相關的指標。"""
        if not self.metrics_history:
            return
        
        # 計算基本指標
        total_rounds = self.round
        total_malicious = len([m for m in self.metrics_history if m.get("malicious", False)])
        total_challenges = len([m for m in self.metrics_history if m.get("challenged", False)])
        successful_challenges = len([m for m in self.metrics_history if m.get("challenge_success", False)])
        
        # 計算率
        challenge_success_rate = successful_challenges / total_challenges if total_challenges > 0 else 0
        malicious_detection_rate = successful_challenges / total_malicious if total_malicious > 0 else 0
        
        logger.info(f"\n----- 研究指標 (輪 {self.round}) -----")
        logger.info(f"總輪數: {total_rounds}")
        logger.info(f"總惡意聚合: {total_malicious}")
        logger.info(f"總挑戰次數: {total_challenges}")
        logger.info(f"成功挑戰: {successful_challenges}")
        logger.info(f"挑戰成功率: {challenge_success_rate:.2f}")
        logger.info(f"惡意檢測率: {malicious_detection_rate:.2f}")
        logger.info(f"被剔除的聚合器: {self.excluded_aggregators}")
        logger.info(f"剩餘聚合器數量: {self.num_aggregators - len(self.excluded_aggregators)}")
        logger.info(f"總回滾次數: {self.total_rollbacks}")
        logger.info(f"攻擊情況下的總輪數: {self.total_rounds_with_attack}")
        logger.info("--------------------------------------------\n")