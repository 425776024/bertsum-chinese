# 抽取式文本摘要模型bertsum，接口部署
（config.py下配置，放好模型）
运行web_main.py，启动http接口

````
request:{
 url : ip/api_summary
 type: post
 doc : '原始文本'
}

return:{
    摘要文本
}
```