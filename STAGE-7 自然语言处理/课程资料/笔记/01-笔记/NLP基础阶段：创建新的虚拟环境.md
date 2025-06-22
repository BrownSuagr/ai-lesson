## NLP基础阶段：创建新的虚拟环境

- 第一步：查看有多少个虚拟环境

  ```properties
  conda env list
  ```

- 第二步：创建一个新的虚拟环境，起个名字：nlpbase

  ```properties
  打开anconda prompt终端，输入命令: conda create -n nlpbase python=3.10
  ```

- 第三步：激活nlpbase虚拟环境

  ```properties
  输入命令:conda activate nlpbase
  ```

- 第四步：安装第三方库

  ```properties
  输入命令: pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple/
  或者输入命令: pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple/  包名
  ```
  ```pro
  pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple/
  pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple/
  pip install joblib -i https://pypi.tuna.tsinghua.edu.cn/simple/
  ```

- 第五步：在pycharm中选择nlpbase虚拟环境

