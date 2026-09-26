# 第三方资源

`可交互的可视化大屏/vendor/` 保留下载的原始脚本，供本地离线展示使用。

| 组件 | 版本 | 来源 | 授权信息 |
| --- | --- | --- | --- |
| Apache ECharts | 5.4.3 | https://www.npmjs.com/package/echarts/v/5.4.3 | Apache-2.0，随附 LICENSE、NOTICE 和 d3 声明 |
| echarts-wordcloud | 2.1.0 | https://www.npmjs.com/package/echarts-wordcloud/v/2.1.0 | npm 包元数据声明 ISC，随附原 package.json |
| 内嵌 wordcloud2.js | 随上述词云包分发 | https://github.com/timdream/wordcloud2.js | MIT，版权注释保留在 `.LICENSE.txt`，完整许可证为 `vendor/wordcloud2-LICENSE.txt` |

下载地址、固定版本和脚本 SHA-256 记录在 `vendor/sources.json`。本仓库不改变这些组件原有的许可条款；词云包没有独立 LICENSE 文件，保留其包元数据及未压缩源码中的原始版权注释。

基础模型由下载命令从 [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) 获取，使用该模型自身的授权条款。权重不随 Git 仓库分发。

`examples/comments.csv` 为本项目新编写的虚构展览评论，不含真实用户、链接、账户标识或原实验评论。标签仅用于演示训练与数据格式，不能用于报告真实场景的模型准确率。
