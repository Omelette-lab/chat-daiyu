import gradio as gr
import lm_api
import sys
import os
from dataclasses import asdict, dataclass
import time
import asyncio  

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/SoVits" % (now_dir))
from api import handle
import logging  

model_lm, tokenizer_lm = lm_api.load_model()
max_length_lm=2048
top_p_lm=0.75
temperature_lm=0.5
generation_config_lm = lm_api.prepare_generation_config(max_length_lm,top_p_lm,temperature_lm) 

user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'

theme = 'ParityError/Anime'



# 配置日志记录器  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  

async def txt_to_audio_async(text):
    return handle(text)
# 音频模块

async def txt_to_audio(text):  
    return await txt_to_audio_async(text)
      


async def daiyu_chat(question: str, chat_history: list = None):
    logging.info("变量 question 的值为: ",question)
    logging.info("变量 question 的类型为: ",type(question))
    if question == None or len(question) < 1:
        return "", chat_history
    try:
        real_prompt = lm_api.combine_history(question,chat_history)
        result = ""
        # 遍历 generate_interactive 函数生成的响应，并将它们添加到列表中  
        for cur_response in lm_api.generate_interactive(  
            model=model_lm,  
            tokenizer=tokenizer_lm,  
            prompt= real_prompt,  
            additional_eos_token_id=92542,  
            **asdict(generation_config_lm),  
        ):  
            result = cur_response
        logging.info("result: ",result)
        ans = result
        audio_file = await txt_to_audio(ans)
        # logging.info("等待30s.....")
        # time.sleep(15)
        logging.info("audio_file: ",audio_file)
        # 聊天函数
        chat_history.append(
            (question, ans)
        )
        return "", chat_history ,audio_file
    except Exception as e:
        return e, chat_history

def daiyu_message(question: str, chat_history: list = None):
    # question = "你是谁？"
    logging.info("变量 question 的值为: ",question)
    logging.info("变量 question 的类型为: ",type(question))
    if question == None or len(question) < 1:
        return "", chat_history
    try:
        real_prompt = lm_api.combine_history(question,chat_history)
        result = ""
        # 遍历 generate_interactive 函数生成的响应，并将它们添加到列表中  
        for cur_response in lm_api.generate_interactive(  
            model=model_lm,  
            tokenizer=tokenizer_lm,  
            prompt= real_prompt,  
            additional_eos_token_id=92542,  
            **asdict(generation_config_lm),  
        ):  
            result = cur_response
        logging.info("result: ",result)
        ans = result
        # 聊天函数
        chat_history.append(
            (question, ans)
        )
        return "", chat_history 
    except Exception as e:
        return e, chat_history



block_1 = gr.Blocks()
with block_1 as demo_1:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown(
                """
                <h1>
                <center>Chat-daiyu</center>
                </h1>
                <center>“故人是谁？”“姑苏林黛玉。”</center>
                """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            with gr.Row():
                audiobot = gr.Audio(
                    type="filepath",
                    interactive=False,
                    autoplay=True
                )
            with gr.Row():
                gr.Image(
                    value="/root/chat-daiyu/src/daiyu.jpg",
                    interactive=False,
                    height="auto",
                    label="daiyu",
                    type="pil"
                )

            with gr.Row():
                max_length_lm = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                top_p_lm = gr.Slider(0, 1, value=0.8, step=0.01,label="Top P", interactive=True)
                temperature_lm = gr.Slider(0, 1.5, value=0.95, step=0.01, label="Temperature", interactive=True)

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=695, show_copy_button=True)
        

    with gr.Row(equal_height=True):
        with gr.Column(scale=8):
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="在此输入聊天内容...")
            with gr.Row():
                # 创建提交按钮。
                submit_btn = gr.Button("发送")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="清除聊天记录")
            

        # 设置按钮的点击事件。当点击时，调用上面定义的函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        submit_btn.click(
            fn=daiyu_chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot,audiobot],
        )








        # submit_btn.click(txt_to_audio,
        #                inputs=[chatbot],
        #                outputs=[audiobot])



#     # gr.Markdown("""🍏与赫萝闲聊时的提示🍎：
#     # <br>
#     # 1. 🎯语音版因为算力限制，运算时间较长(>=20s, <=100s)，请耐心等待🎯
#     # 2. ✨如果希望能够与贤狼赫萝快速沟通，建议使用 Chatty-Chatty 版本(左上角 Tab)✨
#     # 3. 🌠版本虽然有一定鲁棒性，但是限于个人技术，请尽可能使用中文且减少错字🌠
#     # 4. 🆕该版本下，模型对问题的回答会转化为音频，放置于音频输出框内🆕
#     # 5. 🍟由于项目技术实现部分是我个人独自负责，所以模型可能会出现一些小问题，感谢您的指正🍟
#     # <br>
#     # """)

# threads to consume the request
gr.close_all()

# 构建 gradio 对话
block_2 = gr.Blocks()
with block_2 as demo_2:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown(
                """
                <h1>
                <center>Chat-daiyu简洁版</center>
                </h1>
                <center>“故人是谁？”“姑苏林黛玉。”</center>
                """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)

    with gr.Row(equal_height=True):
        with gr.Column(scale=8):
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="在此输入聊天内容...")
            with gr.Row():
                # 创建提交按钮。
                submit_btn = gr.Button("发送")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")

        # 设置按钮的点击事件。当点击时，调用上面定义的函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        submit_btn.click(
            fn=daiyu_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

# #     gr.Markdown("""🍏与赫萝闲聊时的提示🍎：
# #     <br>
# #     1. 🎯语音版因为算力限制，运算时间较长(>=20s, <=100s)，请耐心等待🎯
# #     2. ✨如果希望能够与贤狼赫萝快速沟通，建议使用 Chatty-Chatty 版本(左上角 Tab)✨
# #     3. 🌠版本虽然有一定鲁棒性，但是限于个人技术，请尽可能使用中文且减少错字🌠
# #     4. 🍟由于项目技术实现是我个人独自负责，所以模型实现会出现一些小问题，感谢您的指正🍟
# #     <br>
# #     """)

# threads to consume the request
demo = gr.TabbedInterface(
    [block_1, block_2],
    # ["Voicy_Voicy", "Chatty_Chatty"],
    # [block_1],
    ["Chat-daiyu","Chat-daiyu简洁版"],
    theme=theme
)
# threads to consume the request
gr.close_all()
# # 针对 Gradio的美化

# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch(share=True, server_port=7860)

