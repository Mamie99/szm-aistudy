from typing import Annotated, TypedDict, List
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END

from tools.hr_tools import get_employee_profile, get_leave_balance, generate_employment_certificate
from agent.rag_pipelline2 import search_hr_policy

# 1.定义全局共享状态(state)
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_uid: str
    loop_state: int

# 2.初始化 llm 与工具绑定
llm = init_chat_model(
    model=os.getenv('DASHSCOPE_CHAT_MODEL_NAME'),
    base_url=os.getenv('DASHSCOPE_BASE_URL'),
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model_provider='openai',
    temperature=0.0,
)

tools = [get_employee_profile, get_leave_balance ,generate_employment_certificate, search_hr_policy]
llm_with_tools = llm.bind_tools(tools)
tools_node = ToolNode(tools)

# 3.定义执行节点
def chatbot_node(state: AgentState):
    """「执行者节点」意图理解、工具调用与内容生成 """
    messages = state.get('messages', [])

    # 首轮对话注入System Prompt
    if len(messages) == 1:
        system_msg = SystemMessage(
            content=f"你是飞羽科技的高级HR智能助理。\n"
                    f"当前提问员工 UID 为{state.get("current_uid")}。\n "
                    f"请务必先调用 get_employee_profile 获取该员工的工作属性，再回答具体问题。\n"
                    f"必须基于工具返回的事实，绝对不能编造数字或条件!"
        )
        messages = [system_msg] + messages

    response = llm_with_tools.invoke(messages)
    return {'messages': [response], 'loop_state': state.get('loop_state', 0) + 1}

class FactCheckResult(BaseModel):
    is_pass: bool = Field(description='如果AI的回答完全忠于知识库原文输出True，捏造了数字或者政策则输出False')
    feedback: str = Field(description='如果 False，指出造假点；如果 True，输出 PASS')



# 从对话历史里找出 RAG 检索返回的知识库原文
# 调另一个 LLM 把"知识库原文"和"AI回答"逐字比对
# 如果发现 AI 编造了数据 → 打回去重写；没编造 → 放行
def fact_checker_node(state: AgentState):
    """「审计者节点」后置事实检验 (Self-Reflection) """
    messages = state['messages']
    last_message = messages[-1]

    # 逆向查找 RAG 召回的原文
    rag_context = ''
    for msg in reversed(messages):
        if getattr(msg, 'name', '') == 'search_hr_policy':
            rag_context = msg.content
            break

    # 若未调用知识库，直接放行
    if not rag_context:
        return {'messages': []}

    print('\n「审计者介入」正在核查生成内容是否包含幻觉..')
    check_llm = init_chat_model(
        model=os.getenv('DASHSCOPE_CHAT_MODEL_NAME'),
        base_url=os.getenv('DASHSCOPE_BASE_URL'),
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model_provider='openai',
        temperature=0.0,
    )
    parser = JsonOutputParser(pydantic_object=FactCheckResult)
    check_prompt = (
        f'你是一个冷酷的合规审计员。对比以下「知识库原文」和「AI生成的恢复」。\n'
        f'「知识库原文」:\n{rag_context}\n'
        f'「AI生成的恢复」:\n{last_message}\n'
        f'严查金额、职级门槛、天数!发现捏造请判 False 并给出修改意见。'
        f'{parser.get_format_instructions()}'   # 要求返回 JSON
    )

    response = check_llm.invoke(check_prompt)

    # 手动解析 JSON
    try:
        result = parser.invoke(response)
        is_pass = result.get('is_pass', True)
        feedback = result.get('feedback', 'Pass')
    except Exception as e:
        print(f'「审计异常」JSON解析失败，默认放行。原因: {e}')
        is_pass = True
        feedback = 'Pass'

    if is_pass:
        print('「审计通过」回答安全，无幻觉')
        return {'messages': []}
    else:
        print(f'「发现幻觉」拦截生成!审计意见: {feedback}')
        correction_msg = HumanMessage(
            content=f'[SYSTEM AUDIT FAILED] 试试错误反馈：{feedback}。请根据知识库原文重写，绝不可包含虚假数据。'
        )
        return {'messages': [correction_msg]}


# 4.定义路由逻辑
def router_after_chatbot(state: AgentState):
    """ Chatbot 输出后的路由判断 """
    last_message = state['messages'][-1]

    if last_message.tool_calls:
        return 'tools'
    else:
        return 'fact_checker'

def router_after_fact_checker(state: AgentState):
    """ 审计完成后的路由判断 """
    last_message = state['messages'][-1]

    if isinstance(last_message, HumanMessage):
        # 防止审计节点和chatbot节点可能在 "幻觉→打回→又幻觉→再打回"的死循环里来回跑
        if state.get('loop_state', 0) > 4:
            print('「强制熔断」反思次数达到上限，放弃纠错')
            return 'end'
        print('「打回重写」图路由指针倒流回chatbot节点...')
        return 'chatbot'
    else:
        return 'end'

# 5.构建状态图
hr_agent_app = (
    StateGraph(AgentState)
    .add_node('chatbot', chatbot_node)
    .add_node('tools', tools_node)
    .add_node('fact_checker', fact_checker_node)

    .add_edge(START, 'chatbot')
    .add_conditional_edges('chatbot', router_after_chatbot,
        { 'tools': 'tools','fact_checker': 'fact_checker',}
    )
    .add_edge('tools', 'chatbot')
    .add_conditional_edges('fact_checker', router_after_fact_checker,
        {'chatbot': 'chatbot','end': END,}
    )
    .compile()
)






