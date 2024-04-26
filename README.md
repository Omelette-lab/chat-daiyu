# chat-daiyu
A digital person who can imitate Daiyu's speech


# 快速开始
将项目clone到本地：
```
git clone https://github.com/Omelette-lab/chat-daiyu.git
```
安装相关依赖：
```
cd chat-daiyu
pip install -r reqirements.txt
```
打开`app.py`，直接运行：
```
cd chat-daiyu
python app.py
```

# 介绍

使用《红楼梦》原著、《87版红楼梦剧本》以及网上一些改编的林黛玉语录作为数据集来源，对InternLM大模型进行微调，让其扮演书中角色林黛玉，是一个既富有创意又技术性强的工作。同样，提取87剧版红楼梦的林黛玉相关音频对GPT-SoVits语音模型进行微调，也是为了在对话中让模型能以林黛玉的声音进行输出。

<br/>

![截图](.\src\model.jpg)

# InternLM大模型微调

## 数据准备：

> 调用chat-gpt3.5的api，给出相应prompt，令其提取出相关数据

- 从《红楼梦》原著中抽取林黛玉的台词和独白。
- 收集87版红楼梦剧本中林黛玉的戏份，并提取其台词。
  ```python
  schema = Object(
      id="script",
      description="将小说改编为剧本，逐行抽取小说中各角色对话的实际内容，注意不要重复提取同一内容",
      attributes=[
          Text(
              id="role",
              description="正在说话的角色",
          ),
          Text(
              id="dialogue",
              description="角色说话的内容",
          )
      ],
      examples=[
          (
              '''
              黛玉再看了一看，冷笑道：“我就知道，别人不挑剩下的，也不给我。替我道谢罢！”周瑞家的听了，一声儿不言语。宝玉便问道：“周姊姊，你作什么到那边去了？”
              ''',
              [
                  {"role": "黛玉", "dialogue": "我就知道，别人不挑剩下的，也不给我。替我道谢罢！"},
                  {"role": "宝玉", "dialogue": "周姊姊，你作什么到那边去了？"}
              ],
          ),
          (
              '''
              话犹未了，林黛玉已摇摇的走了进来。一见了宝玉，便笑道：“嗳哟，我来的不巧了！”宝玉等忙起身笑让坐。宝钗因笑道：“这话怎么说？”
              ''',
              [
                  {"role": "黛玉", "dialogue": "嗳哟，我来的不巧了！"},
                  {"role": "宝钗", "dialogue": "这话怎么说？"}
              ],
          )
      ],
      many=True,
  )
  ```
- 收集网上改编的林黛玉语录，用于丰富数据集。
  ```python
  schema = Object(
      id="script",
      description="接下来你会收到林黛玉的语录，请你根据收到的语录生成相应的上一句对话人说的话（3-5）个，并按照示例给的格式输出",
      attributes=[
          Text(
              id="role",
              description="正在说话的角色(对话人和黛玉轮流)",
          ),
          Text(
              id="dialogue",
              description="角色说话的内容",
          )
      ],
      examples=[
          (
              '''
              你大抵是倦了，竟回我这般敷衍。
              ''',
              [
                  {"role": "对话人", "dialogue": "嗯"},
                  {"role": "黛玉", "dialogue": "你大抵是倦了，竟回我这般敷衍。"},
                  {"role": "对话人", "dialogue": "哦"},
                  {"role": "对话人", "dialogue": "确实"},
                  {"role": "黛玉", "dialogue": "你大抵是倦了，竟回我这般敷衍。"},
                  {"role": "对话人", "dialogue": "呵呵"},
                  {"role": "黛玉", "dialogue": "你大抵是倦了，竟回我这般敷衍。"},
              ],
          ),
          (
              '''
              好不容易回个消息，难为你费心，哪里就等死我了呢
              ''',
              [
                  {"role": "对话人", "dialogue": "不好意思，刚刚忙去了，没看到消息"},
                  {"role": "黛玉", "dialogue": "好不容易回个消息，难为你费心，哪里就等死我了呢"},
                  {"role": "对话人", "dialogue": "今天手机没电关机了，没看到你的消息"},
                  {"role": "黛玉", "dialogue": "好不容易回个消息，难为你费心，哪里就等死我了呢"},
                  {"role": "对话人", "dialogue": "今天出门忘带手机了，所以才回你消息"},
                  {"role": "黛玉", "dialogue": "好不容易回个消息，难为你费心，哪里就等死我了呢"},
              ],
          )
      ],
      many=True,
  )
  ```
- 数据清洗：清洗数据，去除重复、无关或错误的台词。

## 模型微调：

将提取、清洗、格式化后的数据集用于训练InternLM大模型。chat-黛玉使用的是InternLM 的 1.8B 模型，模型参数量为 1.8B,底座模型为 interlm-chat-1.8b

### XTuner

使用 XTuner 进行微调，具体脚本可参考[internlm2_1_8b_qlora_alpaca_e3](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/personal_assistant_document.md)，该脚本在train文件夹下。脚本内有较为详细的注释。

经过微调后，模型能够生成与林黛玉风格相符的文本，包括她的机智、敏感、忧郁等特质。当用户与模型对话时，模型能够以林黛玉的口吻进行回复，使对话更具代入感。

![截图](.\src\web_demo1.png)

![截图](.\src\web_demo2.png)

<br/>

# GPT-SoVits语音模型微调

## 数据处理

### 音频提取

- 从87版红楼梦中提取林黛玉的相关音频片段。确保提取的音频质量清晰，且能够准确反映林黛玉的声音特点。

### 音频处理

- 对提取的音频进行预处理，如降噪、标准化等。
- 将音频转换为模型训练所需的格式.wav。

## 模型微调

使用处理后的音频数据对GPT-SoVits语音模型进行微调。在微调过程中，关注模型对林黛玉声音特征的捕捉和再现能力，并调整数据集。

经过对模型GPT-SoVits后，能够生成与87版红楼梦中林黛玉声音相似的语音。当用户与模型进行对话时，模型能够以林黛玉的声音进行回复，增强对话的沉浸感。

# 整合与交互

## 模型整合

将微调后的InternLM文本生成模型和GPT-SoVits语音模型进行整合，编写相关调用函数，引入异步函数，确保两个模型能够无缝对接，实现文本到语音的转换。

### 前后端交互：

设计并实现一个前端界面，允许用户与模型进行对话。前端界面能够接收用户的文本输入，并调用模型生成林黛玉风格的回复和语音输出。

用户通过前端界面与模型进行对话时，能够感受到与书中角色林黛玉的深入交流。模型生成的回复不仅具有林黛玉的说话风格，而且能够以她的声音进行输出，为用户带来沉浸式的阅读体验。通过这样的微调与整合工作，不仅可以展示技术的先进性，还能够让更多的人通过现代技术更深入地了解和体验《红楼梦》这部经典文学作品的魅力。

# 测评与量化

## OpneCompass 评测

- 安装 OpenCompass

```powershell
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

- 下载解压数据集

```powershell
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
```

- 评测启动！

```powershell
python run.py \
    --datasets ceval_gen \
    --hf-path /root/daiyu-chat/final_model \
    --tokenizer-path /root/daiyu-chat/final_model \
    --tokenizer-kwargs padding_side='left' truncation='left'     trust_remote_code=True \
    --model-kwargs device_map='auto' trust_remote_code=True \
    --max-seq-len 2048 \
    --max-out-len 16 \
    --batch-size 2  \
    --num-gpus 1 \
    --debug
```

量化结果保存在 [results](./results)。
