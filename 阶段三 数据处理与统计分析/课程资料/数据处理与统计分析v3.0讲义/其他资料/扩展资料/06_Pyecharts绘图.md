# Echarts和Pyecharts

## 学习目标

- 掌握pyecharts绘图

## 1 Echarts 和 Pyecharts简介

**Echarts简介**

- ECharts 是一个使用 JavaScript 实现的开源可视化库，涵盖各行业图表，满足各种需求。
- ECharts 遵循 Apache-2.0 开源协议，免费商用。
- ECharts 兼容当前绝大部分浏览器（IE8/9/10/11，Chrome，Firefox，Safari等）及兼容多种设备，可随时随地任性展示。

**pyecharts简介**

- pyechart是一个用书生成Echarts图表的Python开源类库
- 使用echart的绘图效果比matplotlib等更加炫酷

## 2 Pyecharts绘图案例

- 由于前面的内容基本已经介绍了常用可视化图标，和各自的特点，下面通过一个案例来介绍Pyecharts的使用
- 案例中使用的pyecharts版本是 1.6.0 ，pyecharts 0.X版本和 1.X版本 API变化较大，不能向下兼容，网上查资料的时候需要注意

### 2.1 案例数据说明

- 案例使用从招聘网站（拉钩、智联、前程无忧、猎聘）上爬取的一周之内数据分析在招岗位数据
- 分析目标：哪些公司在招聘数据分析，哪些城市数据分析的需求大，不同城市数据分析的薪资情况，数据分析对工作年限的要求，数据分析对学历的要求
- 加载数据

```python
import pandas as pd
# 加载数据
job = pd.read_csv('data/data_analysis_job.csv')
# 最近一周 python相关工作数量
job.shape
```

><font color='red'>显示结果：</font>
>
>```
>(7926, 9)
>```

```python
job.head()
```

><font color='red'>显示结果：</font>
>
>|      |                     company_name |                                               url |         job_name | city |     salary | experience |                                   company_area | company_size |                                                  description |
>| ---: | -------------------------------: | ------------------------------------------------: | ---------------: | ---: | ---------: | ---------: | ---------------------------------------------: | -----------: | -----------------------------------------------------------: |
>|    0 |         北京盛业恒泰投资有限公司 | https://jobs.51job.com/beijing-cyq/104869885.h... |   金融数据分析师 | 北京 | 6000-18000 |        NaN |  金融/投资/证券,专业服务(咨询、人力资源、财会) |   500-1000人 | 北京盛业恒泰投资有限公司，注册资金1000万人民币，是一家一站式金融服务的大型企业，旗下有在... |
>|    1 | 第一象限市场咨询（北京）有限公司 | https://jobs.51job.com/beijing/122888080.html?... |       数据分析师 | 北京 |  5000-8000 |    1年经验 | 专业服务(咨询、人力资源、财会),互联网/电子商务 |     少于50人 | 工作职责：1、协同团队完成客户委托的市场研究项目，包括项目的设计、执行、分析、报告撰写和汇报... |
>|    2 |     北京超思电子技术有限责任公司 | https://jobs.51job.com/beijing/123358754.html?... | 市场数据分析专员 | 北京 | 7000-10000 |    1年经验 |         电子技术/半导体/集成电路,医疗设备/器械 |   500-1000人 | 岗位职责：1、业务数据分析l编写月度、季度、年度业务数据分析报告，开展各类与业务销售数据分析... |
>|    3 |         北京麦优尚品科贸有限公司 | https://jobs.51job.com/beijing-cyq/79448137.ht... |       数据分析员 | 北京 | 8000-10000 |    1年经验 | 快速消费品(食品、饮料、化妆品),互联网/电子商务 |     50-150人 | 1、根据运营总监安排，实施具体数据分析工作。2、负责数据的审核工作，确保数据的准确性。3、总... |
>|    4 |             北京磐程科技有限公司 |                                       https://... |   金融数据分析师 | 北京 | 8000-10000 |    1年经验 |                                 金融/投资/证券 |     50-150人 | 1.人品端正，有责任心，愿与公司一同成长进步；2.以金融行业为最后的事业并为之奋斗一生;3.... |

- 查看数据字段

```python
job.columns
```

><font color='red'>显示结果：</font>
>
>```
>Index(['company_name', 'url', 'job_name', 'city', 'salary', 'experience',
>       'company_area', 'company_size', 'description'],
>      dtype='object')
>```

- company_name:公司名字，url，工作的网址，job_name : 工作名字，city：工作所在城市，salary：工资 experience：经验，company_area：公司所在行业，company_size：公司规模，description：工作描述

### 2.2 哪些城市在数据分析机会多

- 按城市分组

```python
city_job_list = job.groupby('city')['url'].count().sort_values(ascending = False)
city_job_top20 = city_job_list.head(20)
city_job_top20
```

><font color='red'>显示结果：</font>
>
>```shell
>city
>北京    1608
>上海    1176
>广州    1033
>深圳     820
>杭州     395
>成都     348
>武汉     320
>西安     248
>南京     222
>苏州     150
>合肥     140
>重庆     130
>长沙     128
>济南     123
>郑州     114
>太原     113
>大连      94
>青岛      94
>东莞      83
>无锡      81
>Name: url, dtype: int64
>```

- pyechart 绘制柱状图

```python
from pyecharts import options as opts
from pyecharts.charts import Bar
c = (
    Bar() #创建柱状图
    .add_xaxis(city_job_top20.index.tolist()) #添加x轴数据
    .add_yaxis('数据分析就业岗位数量', city_job_top20.values.tolist())#添加y轴数据
    .set_global_opts( #设置全局参数
        title_opts=opts.TitleOpts(title='一周内Python就业岗位数量'), #设置标题
        datazoom_opts=opts.DataZoomOpts(),#添加缩放条
    )
)
c.render_notebook() # 在juypter notebook中显示
```

><font color='red'>显示结果：</font>
>
>![image-20200821200221458](img/echarts_bar.png)

- 从结果中可以看出，北京上海广州深圳等一线城市，对数据分析的需求最为旺盛

### 2.3 哪些公司在招聘数据分析师

- 根据公司名字对数据进行分组

```python
company_list = job.groupby('company_name')['url'].count().sort_values(ascending = False)
company_list = company_list.head(100)
company_list
```

><font color='red'>显示结果：</font>
>
>```shell
>company_name
>北京字节跳动科技有限公司        213
>武汉益越商务信息咨询有限公司       85
>北京融汇天诚投资管理有限公司       81
>字节跳动                 67
>腾讯科技（深圳）有限公司         43
>                   ... 
>北京华融泰克科技有限公司          7
>广发银行股份有限公司信用卡中心       7
>软通动力信息技术(集团)有限公司      7
>中国平安人寿保险股份有限公司        7
>墨博（湖南）文化传媒有限公司        7
>Name: url, Length: 100, dtype: int64
>```

- pyecharts绘制词云图

```python
from pyecharts.charts import WordCloud
c = (
    WordCloud()#创建词云图对象
    .add(series_name='哪些公司在招聘数据分析程序员', #添加标题
         data_pair=list(zip(company_list.index.tolist(),company_list.values.tolist())), 
         word_size_range=[6, 40])#指定文字大小，注意如果字体太大可能显示不全
    .set_global_opts(#设置全局参数 标题，字号
        title_opts=opts.TitleOpts(title='哪些公司在招聘数据分析程序员', 
                                  title_textstyle_opts=opts.TextStyleOpts(font_size=23))
    )
)
```

```python
c.render_notebook()
```

><font color='red'>显示结果：</font>
>
>![image-20200821203345286](img/echarts_wordcloud.png)

### 2.4 岗位数量与平均起薪分布

- 数据清洗，提取起薪

  - 原始数据中 大多数薪资都以'6000-18000' 形式显示，需要对数据进行处理，提取出起薪，可以使用正则表达式，配合apply自定义函数，将工作起薪提取出来

  ```python
  # 数据清洗 提取起薪
  import re
  def get_salary_down(x):
      if ('面议' in x) or ('小时' in x) or ('以' in x):
          return 0
      elif '元' in x:
          return int(re.search(pattern="([0-9]+)-([0-9]+)元/月", string=x).group(1))
      elif '-' in x:
          return int(re.search(pattern="([0-9]+)-([0-9]+)", string=x).group(1))
      else:
          return int(x)
  ```

  - 提取起薪数据之前，首先处理缺失值，将缺失数据去掉并提取起薪

  ```pyth
  job.dropna(subset = ['salary'],inplace = True)
  job['salary_down'] = job.salary.apply(get_salary_down)
  job['salary_down'].value_counts().sort_index()
  ```

  ><font color='red'>显示结果：</font>
  >
  >```shell
  >0         141
  >1000        5
  >1500        8
  >1800        1
  >2000       79
  >         ... 
  >60000       3
  >62500       1
  >70000       2
  >80000       1
  >100000      1
  >Name: salary_down, Length: 92, dtype: int64
  >```

- 数据异常值处理

  - 从处理的起薪中发现，有一些异常数据
    - 特别低的：小于3000
    - 特别高的：大于80000
  - 异常值处理：直接删除

  ```python
  job_salary = job[job['salary_down']>3000]
  job_salary = job_salary[job_salary['salary_down']<80000]
  ```

- 绘制气泡图数据准备

  - 将所有城市的薪资和就业岗位数合并到一起

  ```python
  def prepare_data(data):
      # 取出各城市有多少岗位
      temp1 = data.groupby('city')['url'].count().sort_values(ascending = False)
      # 计算出各城市岗位的平均起薪
      temp = data.groupby('city')['salary_down'].mean().sort_values(ascending = False)
      # 合并数据
      temp2 = pd.concat([temp,temp1],axis = 1)
      temp2 = temp2.reset_index()
      return temp2
  ```

  - 处理数据

  ```python
  salary_data = prepare_data(job)
  salary_data.columns = ['city','salary_down','job_count']
  salary_data.head()
  ```

  ><font color='red'>显示结果：</font>
  >
  >|      | city |  salary_down | job_count |
  >| ---: | ---: | -----------: | --------: |
  >|    0 | 北京 | 13882.837209 |      1505 |
  >|    1 | 上海 | 10795.126005 |      1119 |
  >|    2 | 深圳 | 10305.392258 |       775 |
  >|    3 | 杭州 |  9756.156425 |       358 |
  >|    4 | 广州 |  7819.732283 |       889 |

- 气泡图展示数据

```python
from pyecharts.charts import Scatter
from pyecharts.commons.utils import JsCode

c = (
    Scatter() #创建散点图对象
    .add_xaxis(salary_data.salary_down.astype(int))#添加x周数据（薪资）
    .add_yaxis(
        "数据分析岗位数量", #y轴数据说明
        [list(z) for z in zip(salary_data.job_count, salary_data.city)],#Y轴数据，岗位，城市
        label_opts=opts.LabelOpts(#Js代码控制气泡显示提示文字
            formatter=JsCode(
                "function(params){return params.value[2]}" #提示
            )
        ),
    )
    .set_global_opts(#全局变量
        title_opts=opts.TitleOpts(title="数据分析就业岗位数量与平均起薪"),#设置标题
        tooltip_opts=opts.TooltipOpts(#Js代码控制气泡弹窗提示文字
            formatter=JsCode(
                "function (params) {return params.value[2]+ '平均薪资：'+params.value[0]}"
            )
        ),
        visualmap_opts=opts.VisualMapOpts(#控制
            type_="size", max_=1500, min_=200, dimension=1
        ),
        xaxis_opts=opts.AxisOpts(min_=6000,name='平均起薪'),#设置X轴起始值，X轴名字
        yaxis_opts=opts.AxisOpts(min_=300,max_=1550,name='岗位数量'),#设置Y轴起始值，Y轴名字
    )
)
```

```python
c.render_notebook()
```

><font color='red'>显示结果：</font>
>
>![image-20200821212019182](img/echarts_bubble.png)

### 2.5 工作经验需求分析

- 工作经验数据清洗

```python
job['experience'].value_counts()
```

><font color='red'>显示结果：</font>
>
>```shell
>1-3年       1197
>1年经验        897
>不限          854
>3-5年        763
>2年经验        555
>经验3-5年      519
>经验不限        483
>3-4年经验      438
>无需经验        387
>经验1-3年      276
>5-10年       206
>经验5-10年     155
>无经验         130
>5-7年经验      103
>1年以下         97
>经验应届毕业生      95
>10年以上        16
>经验1年以下       15
>8-9年经验        9
>一年以下          8
>经验10年以上       3
>10年以上经验       2
>Name: experience, dtype: int64
>```
- 从上面结果中看出，工作经验的描述不尽相同，需要处理成统一格式

```python
job.experience.fillna('未知',inplace = True)
def process_experience(x):
    if x in ['1-3年','2年经验','经验1-3年']:
        return '1-3年'
    elif x in ['3-5年','经验3-5年','3-4年经验']:
        return '3-5年'
    elif x in ['1年经验','1年以下','经验1年以下','一年以下','经验应届毕业生','不限','经验不限','无需经验','无经验']:
        return '一年以下/应届生/经验不限'
    elif x in ['5-10年','经验5-10年','5-7年经验','8-9年经验','10年以上经验','10年以上','经验10年以上']:
        return '5年以上'
    else:
        return x
    
job['exp'] = job.experience.apply(process_experience)
job['exp'].value_counts()
```

><font color='red'>显示结果：</font>
>
>```shell
>一年以下/应届生/经验不限    2966
>1-3年             2028
>3-5年             1720
>未知                718
>5年以上              494
>Name: exp, dtype: int64
>```

- 圆环图展示结果

```python
from pyecharts.charts import Pie
c = (
    Pie()
    .add(
        series_name="经验要求",
        data_pair=[list(z) for z in zip(
            job['exp'].value_counts().index.tolist(),# 准备数据
            job['exp'].value_counts().values.tolist()
        )],
        radius=["50%", "70%"],#圆环图，大环小环的半径大小
        label_opts=opts.LabelOpts(is_show=False, position="center"),
    )#设置图例位置
    .set_global_opts(
        title_opts=opts.TitleOpts(title="数据分析工作经验要求"),
        legend_opts=opts.LegendOpts(pos_left="right", orient="vertical"))
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(#鼠标滑过之后弹出文字格式
            trigger="item", 
            formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
    )
)
```

```python
c.render_notebook()
```

><font color='red'>显示结果：</font>
>
>![image-20200821214059895](img/echarts_dounut.png)



## 小结

- echarts是基于js的开源可视化库，pyecharts是echarts的python封装，利用pyecharts可以绘制具备交互性的炫酷图形
- pyecharts 1.*版本的绘图api还是具有一定规律的
  - Pie(), Bar() .... 创建绘图对象
  - .add() /add_xaxis,add_yaxis添加数据
  - .set_global_opts()  设置全局参数
  - render_notebook()在notebook中绘制、render()生成文件
- 更加详细的API可以参考pyecharts的[官方文档](https://pyecharts.org/#/zh-cn/intro)和[案例](https://gallery.pyecharts.org/#/README)

