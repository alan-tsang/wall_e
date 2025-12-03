from datetime import datetime
from .base_callback import BaseCallBack
from ..logging.logger import Logger
from ..common.registry import registry


@registry.register_callback("summary")
class SummaryCallBack(BaseCallBack):
    def __init__(self, runner):
        super().__init__(runner)
        self._epoch_start_time = None
        self._epoch_end_time = None
        self._valid_start_time = None
        self._valid_end_time = None
        self._test_start_time = None
        self._test_end_time = None
    
    def before_running_epoch(self):
        """训练epoch开始前记录时间"""
        self._epoch_start_time = datetime.now()
    
    def after_running_epoch(self):
        """训练epoch结束后记录摘要"""
        self._epoch_end_time = datetime.now()
        self._log_summary("Epoch", self._epoch_start_time, self._epoch_end_time)
    
    def before_valid(self):
        """验证阶段开始前记录时间"""
        self._valid_start_time = datetime.now()
    
    def after_valid(self):
        """验证阶段结束后记录摘要"""
        self._valid_end_time = datetime.now()
        self._log_summary("Validation", self._valid_start_time, self._valid_end_time)
    
    def before_test(self):
        """测试阶段开始前记录时间"""
        self._test_start_time = datetime.now()
    
    def after_test(self):
        """测试阶段结束后记录摘要"""
        self._test_end_time = datetime.now()
        self._log_summary("Test", self._test_start_time, self._test_end_time)
    
    def _log_summary(self, phase_name, start_time, end_time):
        """
        通用的摘要日志记录方法
        
        Args:
            phase_name: 阶段名称 (Epoch/Validation/Test)
            start_time: 开始时间
            end_time: 结束时间
        """
        if start_time is None or end_time is None:
            return
        
        elapsed_time = end_time - start_time
        elapsed_seconds = elapsed_time.total_seconds()
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02}:{int(minutes):02}:{seconds:.3f}"
        
        # 获取指标信息
        metrics = registry.get('metric')
        if metrics:
            info_items = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        else:
            info_items = "无"
        
        # 根据不同阶段输出不同格式的日志
        if phase_name == "Epoch":
            self.runner.logger.info(
                f"Epoch {self.runner.state.current_epoch + 1} 用时: {time_str}s 总结：{info_items}\n-----------"
            )
        elif phase_name == "Validation":
            self.runner.logger.info(
                f"验证阶段 用时: {time_str}s 总结：{info_items}\n-----------"
            )
        elif phase_name == "Test":
            self.runner.logger.info(
                f"测试阶段 用时: {time_str}s 总结：{info_items}\n-----------"
            )
            