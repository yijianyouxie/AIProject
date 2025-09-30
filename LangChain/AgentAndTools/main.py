
import sys
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from ModelManager import loadModel
from AgentAndTools import DataAnalysisTools


def main():
    csv_path = './sales_data.csv'
    data_tools = DataAnalysisTools(csv_path)

    # 根据参数 决定使用本地还是远程模型
    number = 0 # 默认是0，表示使用本地模型
    if len(sys.argv) > 1:
        number = int(sys.argv[1])
    llm = loadModel(number)
    if llm is None:
        print(f"main 模型创建失败。number = {number}")
        return
    # 创建智能体
    agent = initialize_agent(
        tools=data_tools.get_tools(),
        llm= llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True,
        handle_parsing_errors = True,
        max_iterations=5
    )

    print("\n===LangChain 数据分析智能体====")
    print(
        """
        可用的分析类型：
        1.数据概览 - 请给我数据概览
        2.统计分析 - 计算销售额的统计信息
        3.数据过滤 - 显示北部地区的销售数据 或 显示电子类产品
        4.数据可视化 - 创建按地区分布的销售额柱状图
        输入 'quit' 退出程序\n
        """
    )
    while True:
        user_input = input("\n你想分析什么？")
        if user_input.lower() in ['quit']:
            print("再见！")
            break
        try:
            response = agent.invoke(user_input)
            print(f"回答：\n{response}")
        except Exception as e:
            print(f"执行出错。{e}")

if __name__ == "__main__":
    main()