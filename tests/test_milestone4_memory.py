import sys
import io
import threading
from pathlib import Path
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from agent.graph_builder import hr_agent_app

class SessionManager:
    """ 会话生命周期管理器：负责超时控制与事件分发 """
    def __init__(self, timeout_seconds: int=5):
        self.timeout_seconds = timeout_seconds
        self.timer = 0
        self.current_thread_id = None
        self.current_uid = None

    def trigger_summary(self):
        """ 倒计时结束触发的方法：向 Graph 发送指令 """
        if not self.current_thread_id:
            return

        config = {'configurable': {'thread_id': self.current_thread_id}}

        # 构建隐藏的触发指令
        idle_trigger_state = {
            'messages': [HumanMessage('__SYS_IDLE_TIMEOUT__')],
            'current_uid': self.current_uid,
        }

        print(f'「后台守护线程」检测到用户：{self.current_thread_id} 闲置超过 {self.timeout_seconds} 秒，触发自动总结。')

        for event in hr_agent_app.stream(idle_trigger_state, config, stream_mode='values'):
            last_message = event['messages'][-1]
            if last_message.type=='ai' and not last_message.tool_calls:
                print(f'{last_message.content}')

        print('接着继续提问')

    def reset_timer(self):
        """ 重置到时器 """
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout_seconds, self.trigger_summary)
        self.timer.start()

    def chat(self, uid:str, thread_id:str, question:str):
        """ 对外聊天接口 """
        self.current_thread_id = thread_id
        self.current_uid = uid

        # 用户说话了，先停止计时器
        if self.timer:
            self.timer.cancel()
        print(f'「UID」提问：{question}')
        config = {'configurable': {'thread_id': thread_id}}

        state = {
            'messages': [HumanMessage(content=question)],
            'current_uid': uid,
            'loop_step': 0,
        }

        for event in hr_agent_app.stream(state, config, stream_mode='values'):
            last_message = event['messages'][-1]
            if isinstance(last_message, HumanMessage):
                continue
            if last_message.type == 'ai' and not last_message.tool_calls:
                print(f'「AI答复」{last_message.content}')

        # 聊天结束，重新开始倒计时
        self.reset_timer()

if __name__ == '__main__':
    print('==========里程碑4：多轮记忆与异步超时总结记录============')
    session = SessionManager(timeout_seconds=30)  # 设定超时时间为 30s

    # 模拟用户连续对话两轮，第一轮
    session.chat(uid='1001', thread_id='session_1001_a', question='你好，我是张三。')

    import time
    time.sleep(5)  # 模拟用户思考5秒

    # 第二轮
    session.chat(uid='1001', thread_id='session_1001_a', question='我还有多少天年假？')
    print('聊天结束，用户离开电脑，开始测试 30s 后闲置自动总结，请不要操作...')

    try:
        time.sleep(60)      # 确保能看到总结输出
    except KeyboardInterrupt:
        pass
    finally:
        if session.timer:
            session.timer.cancel()
        print('===== 测试结束 =====')


