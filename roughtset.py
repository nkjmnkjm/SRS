import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from copy import deepcopy
import numpy as np
import pandas as pd
import xlrd
from pprint import pprint
from itertools import combinations, permutations
import random

class roughset():

    def __init__(self,train_data,train_target):
        self.train = train_data
        self.target = train_target

    def basic_set(self,df):
        basic = {}
        for i in df.drop_duplicates().values.tolist():
            basic[str(i)] = []
            for j, k in enumerate(df.values.tolist()):
                if k == i:
                    basic[str(i)].append(j)
        return basic

    def divide(self):
        matrix = pd.concat([self.train,self.target],axis=1).values
        len_attribute = len(matrix[0]) - 1
        matrix_delete = []
        # 决策表分解
        mistake = []
        a = len(matrix)
        for j in range(len(matrix)):
            for k in range(j, len(matrix)):
                if ((matrix[j][0:len_attribute] == matrix[k][0:len_attribute]).all() and matrix[j][len_attribute] !=matrix[k][len_attribute]):
                    matrix_delete.append(list(matrix[k]))
                    mistake.append(k)
        if mistake!=[]:
            matrix = np.delete(matrix, mistake, axis=0)
        # if (len(matrix_delete)):
        #     print('矛盾:', matrix_delete)
        # else:
        #     print('不存在矛盾数据！')
        #     print('-------------------------------------------')
        # if (len(matrix)):
        #     print('完全一致表:')
        #     pprint(matrix)
        # else:
        #     print('不存在完全一致表:')
        #
        # if (len(matrix_delete)):
        #     print('完全不一致表:')
        #     print(matrix_delete)
        self.matrix = matrix

    def yilaidu_fun(self,k):
        excel_temp = list(combinations(self.excel, k))  # 排列组合数
        title_temp = list(combinations(self.title, k))
        for i in range(len(title_temp)):
            excel_temp[i] = list(excel_temp[i])
            title_temp[i] = list(title_temp[i])
        # print('title_temp:',title_temp)
        # print(excel_temp)
        a = []
        b = []
        c = []
        for i in range(len(excel_temp)):
            temp = self.intersection(excel_temp[i])
            # print(temp)
            # print(title_temp[i], end=' ')
            # print('依赖度:', self.dependence_degree(temp))
            # if (self.dependence_degree(temp) == 1):
            #     a.append(set(title_temp[i]))
            # else:
            #     b.append(set(title_temp[i]))
            #     c.append(self.dependence_degree(temp))
            a.append(set(title_temp[i]))
            b.append(self.dependence_degree(temp))
        # print(a)
        return a,b

    def dependence_degree(self,x_list):
        count = 0
        for i in x_list:
            for j in self.ybasic_list:
                if (set(i) <= set(j)):
                    count = count + len(i)
        degree = round(count / len(self.train), 4)
        return degree

    def intersection_2(self,someone_excel):
        x_list = someone_excel[0]
        y_list = someone_excel[1]
        # print(x_list)
        # print(y_list)
        x_set = [[]] * len(x_list)
        for i in range(len(x_list)):
            x_set[i] = set(x_list[i])

        y_set = [[]] * len(y_list)
        for i in range(len(y_list)):
            y_set[i] = set(y_list[i])

        a = []
        for i in range(len(x_list)):
            for j in range(len(y_list)):
                if (x_set[i] & y_set[j]):
                    a.append(x_set[i] & y_set[j])
        return a

    def intersection(self,excel_temp):
        if (len(excel_temp) > 1):
            a = self.intersection_2([excel_temp[0], excel_temp[1]])
            for k in range(2, len(excel_temp)):
                a = self.intersection([a, excel_temp[k]])
        else:
            a = excel_temp[0]
        return a

    def deal(self):
        self.excel=[]
        self.title = self.train.columns.values.tolist()
        self.divide()
        matrix_T = self.matrix.T
        for i in range(self.train.shape[1]):
            a = self.train.loc[:,self.title[i]]
            basicset= self.basic_set(self.train.loc[:,self.title[i]])
            basiclist = sorted([v for k,v in basicset.items()])
            self.excel.append(basiclist)
        self.ybasic_list = sorted([v for k,v in self.basic_set(self.target).items()])
        yilaidu = []
        ratelist = []
        for i in range(1, self.train.shape[1] + 1):
            # a,b,c = self.yilaidu_fun(i)
            # yilaidu.extend(a)
            # for j in b:
            #     yuejian.append(j)
            # for p in c:
            #     yilaidu1.append(p)
            a,b = self.yilaidu_fun(i)
            yilaidu.extend(a)
            ratelist.extend(b)
        #print('依赖度为1的属性有:', yilaidu)
        # print(yilaidu)
        numlist = [i for i, j in enumerate(ratelist) if j == max(ratelist)]
        for i in numlist:
            for j in numlist:
                if (yilaidu[i]>yilaidu[j]):
                    yilaidu[i]=yilaidu[j]
                    num = j
        if len(numlist)==1:
            num = numlist[0]
        for i in range(len(yilaidu)):
            if i not in numlist:
                yilaidu[i] = yilaidu[num]
        # 约简
        # i = 0
        # j = 0
        # k = len(yilaidu)  # k=6
        # for i in range(k):
        #     for j in range(k):
        #         if (yilaidu[i] > yilaidu[j]):  # i更大，应该删除i
        #             yilaidu[i] = yilaidu[j]

        # 去重
        yilaidu_new = []
        for i in yilaidu:
            if i not in yilaidu_new:
                yilaidu_new.append(i)
        for i in range(len(yilaidu_new)):
            yilaidu_new[i] = sorted(list(yilaidu_new[i]))
        result = yilaidu_new

        # 各约简属性的属性值
        matrix_new = []
        for i in range(len(result)):
            matrix_new.append([])
        for i in range(len(result)):
            for j in range(len(result[i])):
                for k in range(len(self.title)):
                    if (result[i][j] == self.title[k]):
                        matrix_new[i].append(list(matrix_T[k]))
        print(result)
        # for i in range(len(matrix_new)):
        #     print('------------------------------------')
        #     print('序号 ', end='')
        #     for j in range(len(result[i])):
        #         print(result[i][j], '', end='')
        #     print('归类')
        #     for j in range(len(matrix_new[0][0])):
        #         print(j + 1, end='    ')
        #         for k in range(len(result[i])):
        #             print(matrix_new[i][k][j], end='  ')
        #         print(' ', self.matrix[j][len(self.matrix[0]) - 1])
        # if (result == []):
        #     print(self.matrix)
        return result

    def deal1(self):
        self.divide()
        self.title = self.train.columns.values.tolist()
        matrix = self.matrix
        matrix_T = matrix.T
        number_sample = len(matrix)  # 样本数量
        number_attribute = len(matrix_T) - 1  # 属性数量
        # 二维列表的创建：
        excel = [[[] for col in range(number_sample)] for row in range(number_sample)]
        pprint(matrix)
        # 比较各样本哪些属性的值不同（只对决策属性不同的个体进行比较）
        for k in range(len(self.title)):  # 属性
            for i in range(number_sample):  # 第几个样本
                for j in range(i, number_sample):
                    if (matrix[i][k] != matrix[j][k] and matrix[i][number_attribute] != matrix[j][number_attribute]):
                        excel[i][j].append(self.title[k])
        for i in range(number_sample):
            for j in range(i, number_sample):
                excel[j][i] = set(excel[i][j])
                excel[i][j] = {}
        # pprint(excel)#excel
        yuejian = []
        for i in range(number_sample):
            for j in range(number_sample):
                if (excel[i][j] and excel[i][j] not in yuejian):
                    yuejian.append(excel[i][j])
        print('约简')
        print(yuejian)
        # 约简
        i = 0
        j = 0
        k = len(yuejian)  # k=6
        for i in range(k):
            for j in range(k):
                if (yuejian[i] > yuejian[j]):  # i更大，应该删除i
                    yuejian[i] = yuejian[j]
                if (yuejian[i] & yuejian[j]):
                    yuejian[i] = yuejian[i] & yuejian[j]
                    yuejian[j] = yuejian[i] & yuejian[j]
        '''
        #print(yuejian)
        #去重
        yuejian_new = []
        for id in yuejian:
            if id not in yuejian_new:
                yuejian_new.append(id)
        yuejian = yuejian_new
        '''
        # print('约简为:',yuejian)
        # print('yuejian:',yuejian)
        # 类似于笛卡儿积
        flag = 0
        result = []
        for i in yuejian:
            if (len(i) > 1):
                flag = 1
        if (flag == 1):  # 将集合分解开，逐个取其与其他集合的并集
            simple = yuejian[0]
            nosimple = deepcopy(yuejian)
            i = 0
            while (i < len(nosimple)):
                if (len(nosimple[i]) == 1):
                    simple = simple | nosimple[i]
                    nosimple.pop(i)
                else:
                    i = i + 1
            for i in range(len(nosimple)):
                nosimple[i] = list(nosimple[i])
            simple = list(simple)

            for i in range(len(nosimple)):
                for j in range(len(nosimple[i])):
                    simple_temp = deepcopy(simple)
                    simple_temp.append(nosimple[i][j])
                    result.append(simple_temp)
        else:
            simple = yuejian[0]
            for i in yuejian:
                simple = simple | i  # 如果只有单元素，则将取其与其他集合的并集
            result.append(list(simple))
        print('jieguo')
        print(result)
        # 约简矩阵的各属性的样本值
        matrix_new = []
        for i in range(len(result)):
            matrix_new.append([])
        for i in range(len(result)):
            for j in range(len(result[i])):
                for k in range(len(self.title)):
                    if (result[i][j] == self.title[k]):
                        matrix_new[i].append(list(matrix_T[k]))
        # 输出
        for i in range(len(matrix_new)):
            print('------------------------------------')
            print('序号 ', end='')
            for j in range(len(result[i])):
                print(result[i][j], '', end='')
            print('归类：')
            for j in range(len(matrix_new[0][0])):
                for k in range(len(result[i])):
                    print(matrix_new[i][k][j], end='   ')
                print(' ', matrix[j][number_attribute])



class roughchange:

    def __init__(self, train_data, train_target, rulist, delta):
        """
        初始化对象参数
        :param train_data: 训练集数据
        :param train_target: 训练集标签
        :param rulist: 已有规则
        :param delta: δ邻域的值，
        """
        self.train_data = train_data
        self.train_target = train_target
        self.delta = delta
        self.rulist  =rulist
        pass

    def compute_corr(self, i, j, tmp_train_data):
        """
        计算相关性
        :param i: 循环第i次
        :param j: 当前特征
        :param tmp_train_data: 临时数据
        :return:
        """
        if i == 0:
            tmp_train_data = self.train_data[:, j]
        # 每一轮更新一个临时数据集，用以测试不同特征下的依赖度的不同
        else:
            tmp_train_data = np.insert(tmp_train_data, tmp_train_data.shape[1], self.train_data[:, j], axis=1)

        # 计算样本的δ邻域
        delta_neighbor_dict = dict()
        for k in range(self.train_data.shape[0]):
            delta_neighbor_list = list()
            for v in range(self.train_data.shape[0]):
                dis = np.sqrt(np.sum((tmp_train_data[k] - tmp_train_data[v]) ** 2))
                if dis <= self.delta:
                    delta_neighbor_list.append(v)
            delta_neighbor_dict.update({k: delta_neighbor_list})

        # 对每个样本判断是否在δ邻域内，是的话更新邻域样本的列表
        sample_list = list()
        for k in range(self.train_data.shape[0]):
            count_issubset = 0
            count = 0
            for v in range(self.train_target.shape[1]):
                if self.train_target[k, v] == 1:
                    count += 1
                    # 每个标签下不同类别及其对应样本索引
                    target_equivalence_class = defaultdict(list)
                    for m, n in [(n, m) for m, n in list(enumerate(self.train_target[:, v]))]:
                        target_equivalence_class[m].append(n)
                    # 前者是否是后者子集
                    if set(delta_neighbor_dict.get(k)).issubset(target_equivalence_class.get(1)):
                        count_issubset += 1
                    else:
                        break
            if count_issubset == count:
                sample_list.append(k)

        # 计算当前特征下的属性依赖度
        corr = len(sample_list) / self.train_data.shape[0]
        return corr

    def rulereduction(self):
        """
            1、一阶段，计算当前特征子集与候选特征子集中任意一个特征后的属性依赖度
            2、属性依赖度的计算，依赖于满足邻域近似条件的样本数目
            3、邻域近似条件指的是当样本的δ邻域内的样本属于样本分类标记中的任意一个标记下的等价类内，该样本满足邻域条件
            4、样本的δ邻域计算是样本与任意个样本进行计算，获取距离，小于δ则加入邻域，
            5、标签等价类是指样本被标记的正类
        :return:
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        x_minmax = min_max_scaler.fit_transform(self.train_data)
        self.train_data = x_minmax
        feature_num = self.train_data.shape[1]
        feature_list_all = []
        corr_first = 0
        old_feature = list(self.train_data)
        print("初始依赖度为0")
        for rulist1 in self.rulist:
            feature_list = []
            for i in range(feature_num):
                tmp_train_data = self.train_data[:, feature_list]
                max_corr = 0
                max_index = 0
                for j in range(rulist1):
                    if j in feature_list:
                        continue
                    j = old_feature.index(rulist1[j])
                    corr = self.compute_corr(i, j, tmp_train_data)
                    print("当前特征数：", len(feature_list), "当前特征：", j, "当前corr：", corr, "最大corr：", corr_first)
                    if max_corr < corr:
                        max_corr = corr
                        max_index = j

                if corr_first < max_corr:
                    corr_first = max_corr
                    feature_list.append(int(max_index))
                else:
                    continue
            feature_list_all.append(feature_list)
        return feature_list_all

    def rulefind(self):
        """
              1、一阶段，计算当前特征子集与候选特征子集中任意一个特征后的属性依赖度
              2、属性依赖度的计算，依赖于满足邻域近似条件的样本数目
              3、邻域近似条件指的是当样本的δ邻域内的样本属于样本分类标记中的任意一个标记下的等价类内，该样本满足邻域条件
              4、样本的δ邻域计算是样本与任意个样本进行计算，获取距离，小于δ则加入邻域，
              5、标签等价类是指样本被标记的正类
          :return:
          """
        min_max_scaler = preprocessing.MinMaxScaler()
        x_minmax = min_max_scaler.fit_transform(self.train_data)
        self.train_data = x_minmax
        feature_num = self.train_data.shape[1]
        feature_list = list()
        corr_first = 0
        print("初始依赖度为0")

        for i in range(feature_num):
            tmp_train_data = self.train_data[:, feature_list]
            max_corr = 0
            max_index = 0
            for j in range(self.train_data.shape[1]):
                if j in feature_list:
                    continue
                corr = self.compute_corr(i, j, tmp_train_data)
                print("当前特征数：", len(feature_list), "当前特征：", j, "当前corr：", corr, "最大corr：", corr_first)
                if max_corr < corr:
                    max_corr = corr
                    max_index = j

            if corr_first < max_corr:
                corr_first = max_corr
                feature_list.append(int(max_index))
            else:
                break

        return feature_list
        pass

        

    

if __name__=='__main__':
    # data = pd.read_csv('1penguins_raw.csv')
    # data = data.fillna(-1)
    #
    #
    # def trans(x):
    #     if x == data['Species'].unique()[0]:
    #         return 0
    #     if x == data['Species'].unique()[1]:
    #         return 1
    #     if x == data['Species'].unique()[2]:
    #         return 2
    #
    #
    # data['Species'] = data['Species'].apply(trans)
    # data_target_part = data[data['Species'].isin([0, 1])][['Species']]
    # data_features_part = data[data['Species'].isin([0, 1])][['Culmen Length (mm)', 'Culmen Depth (mm)',
    #                                                          'Flipper Length (mm)', 'Body Mass (g)']]
    # x_train, x_test, y_train, y_test = train_test_split(
    #     data_features_part,data_target_part, test_size=0.2, random_state=2020)
    # A=ARMLNRS(x_train.values,y_train.values,x_test.values,y_test.values,0.01,0)
    # print('******')
    # a= A.armlnrs()
    # print(a)
    data = pd.read_csv('./断案/0_0.10.csv')
    data1=data.iloc[:,3:12]
    data2 = data.iloc[:,12].to_frame()
    # rough_model = roughset(data1,data2)
    # rough_model.deal()
    from sklearn import linear_model
    data3 = data.iloc[:,2].to_frame()
    model = linear_model.LinearRegression()
    model.fit(data3,data2)
    data4 = model.predict(data3).tolist()
    data2 = data.iloc[:,12].tolist()
    data5=[]
    for i in range(len(data4)):
        a = int(100*(data4[i][0]-data2[i])/data4[i][0])
        data5.append(int(a/10))
    data6 = pd.DataFrame(data5)
    rough_model1 = roughset(data1,data6)
    rough_model1.deal()
    #rough_model1.deal1()