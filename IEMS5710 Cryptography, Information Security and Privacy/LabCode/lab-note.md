### 场景：学生向Blackboard上提交作业

![img_1.png](img_1.png)

#### 具体流程
![img_3.png](img_3.png)
![img_2.png](img_2.png)

##### CUHK(服务端)
> 保持进程监听Student的请求

##### STUDENT（客户端）
> 当输入Student ID后，向CUHK发起请求。

##### BLACKBOARD（服务端）
> 保持进程监听