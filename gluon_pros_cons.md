## 动态图接口gluon优缺点分析

### 优点

\#高效

通过执行```net.hybridize```的方式转为符号式执行，获得性能和更容易移植。0.11版gluon比0.20版torch快20%

节省显存， 命令式开发，符号式部署，结合科研与工程

\#灵活

可以随意的在符号式和命令式之间切换或者做两者的结合，尤其是在自然语言处理和强化学习方面

1 把输入当作tensor处理：

```
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        with self.name_scope():
    def hybrid_forward(self, F, x):
        x = x[:, :, 1:, 1:]
        return x
```

2 使用ndarray写逻辑:

```
from mxnet import nd
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
    def forward(self, x):
        x = nd.slice(x, begin=(None, None, 1, 1), end=(None, None, None, None))
        return x
```

3 使用函数式写（这个名字是我自己给的）:

```
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        with self.name_scope():
    def hybrid_forward(self, F, x):
         x = F.slice(x, begin=(None, None, 1, 1), end=(None, None, None, None))
        return x
```

或者把上面三种方式任意排列组合拼起来用

4 堆积木一样堆积CNN层：

```
class RecMLP(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        with self.name_scope():
            self.net.add(nn.Dense(256, activation="relu"))
            self.net.add(nn.Dense(128, activation="relu"))
            self.net.add(nn.Dense(64, activation="relu"))
    def hybrid_forward(self, F, x):
        return self.net(x))
net = RecMLP()
y = net(data)
```

或者嵌套混合使用Sequential，多数情况下会结合网络的重复结构嵌套使用。

5 gluon兼容符号式写法，给模型传什么就得到什么，如果给模型传tensor，数据会流经每一个地方且可以打断点调试，如果传递符号，则得到图

```
class RecMLP(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        with self.name_scope():
            self.net.add(nn.Dense(256, activation="relu"))
            self.net.add(nn.Dense(128, activation="relu"))
            self.net.add(nn.Dense(64, activation="relu"))
    def hybrid_forward(self, F, x):
        return self.net(x))
#net = RecMLP()
#y = net(data)
net = RecMLP()
net_symbol = net(mx.sym.var("x"))
mx.viz.plot_network(block(net_symbol))
```

6 更细粒度的初始化，定义一个层的时候可以初始化某一层

```gluon.nn.Dense(weight_initializer=mx.init.Normal())```

\#简洁

gluon写的模型更加简洁，自动判断输入长度，指定输出长度就行，这个是优点mxnet的symbol也是支持的，和torch, keras对比的优点

直观易懂，给个nasnet的例子

[用gluon 写的nasnet ](http://code.huawei.com/h00416768/mxnet/blob/nasnet/python/mxnet/gluon/model_zoo/vision/nasnet.py)

[用tensorflow符号式编程写的nasnet](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet)

\#文档强大全面

这个优点是官方给的，写的和torch很像，很多用法文档上都没有，我觉得还是原来的文档比较好用， 源码简洁，开设的深度学习课程是针对gluon的

[英文文档](http://gluon.mxnet.io/)

[中文文档](https://zh.gluon.ai/)

### 缺点

1 成熟度和线上部署方面不足

2 增加了一套接口，拉长了mxnet的战线，熟悉老接口的用户不是很愿意转向新API