# 1 资料
我复现的源码: [https://github.com/Whiffe/time-travel-in-llms/tree/main](https://github.com/Whiffe/time-travel-in-llms/tree/main)

官网源码：[https://github.com/shahriargolchin/time-travel-in-llms](https://github.com/shahriargolchin/time-travel-in-llms)

论文：[https://openreview.net/forum?id=2Rwq6c3tvr](https://openreview.net/forum?id=2Rwq6c3tvr)

论文翻译：[ICLR-2024.Shahriar Golchin.TIME TRAVEL IN LLMS: TRACING DATA CONTAMINATION IN LARGE LANGUAGE](https://blog.csdn.net/WhiffeYF/article/details/142001749)

b站复现视频：[https://www.bilibili.com/video/BV13S1eYqE62/](https://www.bilibili.com/video/BV13S1eYqE62/)

CSDN：[顶会论文复现 time-travel-in-llms, TIME TRAVEL IN LLMS: TRACING DATA CONTAMINATION IN LARGE LANGUAGE MODELS](https://blog.csdn.net/WhiffeYF/article/details/142695383)
# 2 我的总结
整体看来，整个论文就是【提示词工程】，确实十分可悲，现在的高校研究所沦落到了捡大厂的剩菜剩饭吃。

作者采用了两种方式检验污染，也就是两种提示词：Guided与General
采用了三种方式评估污染，两种传统方法（rougeL、Bleurt，我认为这两种不适合对生成式内容进行评估），一种大模型方法（还是大模型的提示词工程来做评估）。

**所以，综上所述，这篇论文就是【提示词工程】的论文，还发的顶会，真是可笑可悲。**



我复现没有用本地部署，直接调用的大模型api接口，评估也用的api接口，rougeL、Bleurt这两种传统方法我觉得也没有实现的必要。

我也没有用GPT3或者4，因为无法直接访问，我就改成了通义千问的api接口。通义千问开发网站：[https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)

# 3 复现源码
开始前需要：

```bash
# 用您的 API Key 代替 YOUR_DASHSCOPE_API_KEY
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"

```
然后执行：

```bash
python  main.py \
        --experiment ./results/gpt4/imdb/test \
        --filepath ./data/imdb/imdb_test.csv \
        --task cls \
        --dataset IMDB \
        --split test \
        --text_column text \
        --label_column label \
        --process_guided_replication  \
        --process_general_replication
```

# 4 结果

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cdc70f979fb74abe8dd586b9eff86519.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a6e233abe4b420ab8d5648664313d53.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a5cce5ab049849dabeb533b5e402ba94.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5112b0db1b0e472cb4100d93ef6ac05f.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f1d1a247cf714d918ed863aa62dcb038.png)
