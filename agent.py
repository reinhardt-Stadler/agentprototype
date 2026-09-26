import ast
import inspect
import os
import re
from string import Template
from typing import List, Callable, Tuple

import click
import platform
from dotenv import load_dotenv
from openai import OpenAI

from prompt_temple import react_system_prompt_temple

class ReActAgent:
    def __init__(self, tools: List[Callable], model: str, project_directory:str):
        self.tools = {func.__name__: func for func in tools}
        self.model = model
        self.project_directory = project_directory
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=ReActAgent.get_api_key(),
        )

    def run(self, user_input:str):
        """跑agent的核心過程"""
        messages = [
            {"role": "system", "content":self.render_system_prompt(react_system_prompt_temple)},
            {"role": "user", "content": f"<question{user_input}</question>"}
        ]
        """這裏首先傳入兩個東西，一個是模型的約束文檔作爲系統縂鋼，一個是用戶的問題，
        然後進入一個while循環，直到模型輸出final answer"""

        while True:
            #首先去請求模型
            content = self.call_model(messages)

            #然後去檢測並提取模型的第一段思考過程
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                print(f"\n\n💭 Thought:{thought}")

            #檢查模型的回應是否是final answer，如果是的話就直接返回
            if "<final_answer>" in content:
                final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                return final_answer.group(1)

            #檢測模型的動作，如果沒有動作就raise一個RuntimeError
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型沒有輸出<action>")
            action = action_match.group(1)
            tool_name, args = self.parse_action(action)
            print(f"\n\n🔧 Action: {tool_name}({', '.join(args)})")

            #儅存在終端命令時，詢問用戶。否則自動執行
            should_continue = input(f"\n\n是否继续？（Y、N）") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消")
                return "操作被用户取消"

            try:
                observition = self.tools[tool_name](*args)
            except Exception as e:
                observition = f"工具执行失败，错误信息：{str(e)}"
            print(f"\n\n📝 Observation: {observition}")
            obs_msg = f"<observition>{observition}</observition>"
            messages.append({"role": "user", "content": obs_msg})
            """此處再次調用messages部分，將觀察結果寫入，並作爲下一輪請求的上下文，
            這樣模型就可以根據觀察結果進行下一步的思考和行動。
            CoT的思想關鍵"""
        

    def get_tool_list(self)-> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature  = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)
    

    def render_system_prompt(self, system_prompt_template:str) ->str:
        """渲染系统提示模板，替换变量"""
        """這裏實際上充當了作爲模型感官的作用，
        將操作系統類型、可用工具列表、可操作文件列表信息傳遞給模型，
        讓模型可以根據這些信息進行思考和行動。
        而且，通過利用substitute方法（他本身的特性，既實現了變量的傳輸，又保證了
        不會被注入某些其他的指令。
        """
        tool_list = self.get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory,f))
            for f in os.listdir(self.project_directory)
        )
        return Template(system_prompt_template).substitute(
            operating_system = self.get_operating_system_name(),
            tool_list= tool_list,
            file_list= file_list
        )

    @staticmethod
    def get_api_key() -> str:
        """从环境变量中获取API Key"""
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY 于.env文件中")
        return api_key


    def call_model(self, messages):
        print("\n\n正在请求模型，请稍等...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        content= response.choices[0].message.content
        messages.append({"role":"assistant", "content":content})
        return content



    def parse_action(self, code_str: str)->Tuple[str, List[str]]:
        """英文是分析動作，實際上就是去解析我的模型輸出，提取出工具名稱和參數列表"""
        """找出回復動作裏面的第一個匹配的函數調用，然後把他手撕了"""
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")
        
        func_name= match.group(1)
        args_str=match.group(2).strip()

        args = []
        current_arg = ""
        in_string = False #是否是動作字符串內部
        string_char = None
        i=0
        parent_depth = 0
        """這裏很精彩，他先定義了一堆變量
        實現了：输入：'name="O\'Brien, Jr.", calc(add(1,2), flag=True)'
        輸出[
                'name="O\'Brien, Jr."',     # 字符串内的逗号和转义引号被正确保留
                'calc(add(1,2), flag=True)' # 嵌套括号内的逗号不被误切
            ]
            的形式。而且由於string_char = char、current_arg += char
            那麽1.记住当前字符串是用哪种引号开头的  
                2.把当前这个字符追加到正在构建的参数字符串里。   
            末尾還利用i+1實現循環控制 ，非常值得藉鑒的思想  """
        while i < len(args_str):
            char = args_str[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    parent_depth += 1
                    current_arg += char
                elif char == ')':
                    parent_depth -= 1
                    current_arg += char
                elif char == ',' and parent_depth == 0:
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None

            i += 1
        
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))

        return func_name, args
        pass

    def _parse_single_arg(self,arg_str:str):
        """解析单个参数"""
        arg_str = arg_str.strip()

        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            inner_str = arg_str[1:-1]
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            return inner_str

        #試用ast.literal_eval來解析其他類型的參數，
        # 這樣可以安全地將字符串轉換為Python對象
        try:
            return ast.literal_eval(arg_str)
        except(SyntaxError, ValueError):
            return arg_str
        #失敗返回原始字符串
        

    def get_operating_system_name(self):
        os_map = {
            "Windows": "Windows",
            "Darwin": "macOS",
            "Linux": "Linux"
        }
        return os_map.get(platform.system(),"Unknown")


def read_file(file_path):
    """读取目标文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def write_to_file(file_path, content):
    """将指定内容写入到目标文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))#把一些文本形式的换行转变为实际可跑的换行
    return "写入成功"

def run_terminal_command(command):
    """執行終端命令"""
    import subprocess
    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr
    
@click.command()
@click.argument('project_directory', type=click.Path(exists=True,file_okay=False, dir_okay=True))
def main(project_directory):
    project_dir = os.path.abspath(project_directory)

    tools = [read_file, write_to_file, run_terminal_command]
    agent = ReActAgent(tools= tools,model="deepseek-flash", project_directory=project_dir)

    task = input("请输入任务：")

    final_answer = agent.run(task)

    print(f"\n\n Final Answer:{final_answer}")

if __name__ == "__main__":
    main()