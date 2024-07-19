import numpy as np
import warnings
import pandas as pd
from itertools import combinations
# from sklearn.metrics.cluster import mutual_info_score
# -*-coding:utf-8-*-
DEBUG = False


def check_df(df, name_col, feature_col, decision_col):
    """
    检查df是否符合要求
    df: pandas.DataFrame
    name_col: str, 样本名称, 必须存在df.columns 且 所有name_col的值都不重复
    feature_col: list, 条件属性, 存在df.columns
    decision_col: str, 决策属性, 存在df.columns
    """
    columns = df.columns
    if DEBUG:
        print(
            f"name_col: {name_col}\n",
            f"feature_col: {feature_col}\n",
            f"decision_col: {decision_col}\n"
        )
    # %% 检查
    # name_col 存在于 columns
    assert name_col in columns, f'{name_col} not in {columns}'
    # 所有name_col的值都不重复
    assert len(df[name_col].unique()) == len(df[name_col]), f'{name_col} has duplicate values'
    # feature_col 存在于 columns
    assert all([col in columns for col in feature_col]), f'{feature_col} not in {columns}'
    # decision_col 存在于 columns
    assert decision_col in columns, f'{decision_col} not in {columns}'


# %% 建立decision dictionary
def create_decision_dict(df, name_col, decision_col):
    """
    建立decision dictionary
    df: pandas.DataFrame
    name_col: str, 样本属性
    decision_col: str, 决策属性

    return: dict, key是decision_values, values是df中decision_col属性为key的name_col属性的值
        例： {0: {1, 2}, 1: {3, 4, 5}}
    """
    decision_values = df[decision_col].unique()
    # 建立一個set，key是decision_values, values是是df中decision_col属性为key的name_col属性的值
    decision_dict = {key: set(df[df[decision_col] == key][name_col]) for key in decision_values}
    return decision_dict


# %% 建立reduct dictionary
def create_reduct_dict_by_row(df, row, name_col, feature_col):
    """
    建立reduct dictionary
    df: pandas.DataFrame
    row: pandas.Series, 要比对的样本
    name_col: str, 样本名称
    feature_col: list, 条件属性
    decision_col: str, 决策属性

    return: dict, key是features, values是df中decision_col是key的name_col的值
        例： {(1,): {1, 2}, (1, 2): {1, 2, 3}}
    """
    reduct_dict = {}
    for num_features in range(1, len(feature_col)):
        for features in combinations(feature_col, num_features):
            df_selected = df  # 取出一份df
            for feature in features:
                value = row[feature]
                df_selected = df_selected[df_selected[feature] == value]  # 依序过滤
            reduct_dict[features] = set(df_selected[name_col])  # 将过滤后的df的name_col的值存入reduct_dict
    return reduct_dict


# %% 过滤reduct dictionary
def filter_reduct_dict_by_row(row, name_col, decision_dict, reduct_dict):
    """
    过滤reduct dictionary
    Args:
        row: pandas.Series, 要比对的样本
        name_col: str, 样本属性
        decision_dict: dict, key是decision_values, values是是df中decision_col属性为key的name_col属性的值
            例：{0: {1, 2}, 1: {3, 4, 5}}
        reduct_dict: dict, key是features, values是是df中decision_col属性为key的name_col属性的值
            例：{('天氣',): {2, 3, 5},
                ('事故情形',): {2},
                ('事故原因',): {2, 5},
                ('天氣', '事故情形'): {2},
                ('天氣', '事故原因'): {2, 5},
                ('事故情形', '事故原因'): {2}}
    Return:
        reduct_result: list, reduct rules
            例：[('事故情形',), ('天氣', '事故情形'), ('事故情形', '事故原因')]
    """
    # 取得decision_dict中value有包含row[name_col]的value
    # 例： {1, 2}
    decision = [value for key, value in decision_dict.items() if row[name_col] in value][0]

    # %%
    # 對所有的reduct_dict的value
    # 分別去計算是否为decision的子集
    # 加入所有的decision的子集到reduct_dict_final
    reduct_result = []
    for key, value in reduct_dict.items():
        if DEBUG:
            print(key, value, value.issubset(decision))
        if value.issubset(decision):
            reduct_result.append(key)
    return reduct_result


# %% 建立reduct rules dataframe
def create_reduct_rules_by_row(df, row, columns, name_col, decision_col, reduct_result, include_empty=False):
    # 针对結果，建立新的dataframe
    df_rule = pd.DataFrame(columns=columns)
    # 先建立一个empty的row
    empty_row = pd.Series([None] * len(columns), index=columns)

    for rule in reduct_result:

        new_row = empty_row.copy()
        for feature in rule:
            new_row[feature] = row[feature]
        new_row[name_col] = row[name_col]
        new_row[decision_col] = row[decision_col]

        df_rule = pd.concat([df_rule, new_row.to_frame().T], ignore_index=True)

    # 若df_rule是空的，则仍然加入一个row
    if df_rule.empty and include_empty:
        new_row = empty_row.copy()
        new_row[name_col] = row[name_col]
        df_rule = pd.concat([df_rule, new_row.to_frame().T], ignore_index=True)

    return df_rule


# %% 建立流程
def create_reduct_rules(df, name_col, feature_col, decision_col, include_empty=False):
    """
    建立流程
    """
    # 检查columns
    check_df(df, name_col, feature_col, decision_col)
    columns = [name_col] + feature_col + [decision_col]

    # 建立决策目标的值的集合
    decision_dict = create_decision_dict(df, name_col, decision_col)

    df_rule = pd.DataFrame(columns=columns)
    # 依照每一个row进行
    for index, row in df.iterrows():
        reduct_dict = create_reduct_dict_by_row(df, row, name_col, feature_col)  # 建立这个row的不同数量特征产生的约简集合
        reduct_result = filter_reduct_dict_by_row(row, name_col, decision_dict, reduct_dict)  # 过滤约简集合，只留下决策目标的子集合
        row_rules = create_reduct_rules_by_row(df, row, columns, name_col, decision_col, reduct_result,
                                               include_empty)  # 建立reduct rules dataframe
        df_rule = pd.concat([df_rule, row_rules], ignore_index=True)
    return df_rule
class cosRoughSet:
    def __init__(self, data, feature_col=None, decision_col=None):
        self.df = data
        self.feature_col = feature_col or list(data.columns[0:-1])
        self.decision_col = decision_col or data.columns[-1]
        self.check_roughset_prerequisites()
        self.featuredf = self.df[self.feature_col]
        self.decisiondf = self.df[self.decision_col]

    def check_roughset_prerequisites(self):
        columns = self.df.columns
        feature_col = self.feature_col
        decision_col = self.decision_col

        assert all([col in columns for col in feature_col]), f'{feature_col} not in {columns}'
        assert all([col in columns for col in decision_col]), f'决策属性不存在：{decision_col} not in {columns}'

    def divdf(self,df,collist):
        divlist = []
        for a, b in df.groupby(collist):
            divlist.append(b.index.tolist())
        divlist = sorted(divlist, key=len, reverse=True)
        ans =[]
        for j in divlist:
            ans.append(set(j))
        return ans
    def getpos(self,collist,dellist):
        pos = []
        for i in collist:
            for j in dellist:
                if i.issubset(j):
                    pos.append(i)
                    continue
        return pos
    def getcos(self,collist,dellist):
        colvec = [len(i) for i in collist]
        delvec = [len(i) for i in dellist]
        while len(colvec)<len(delvec):
            colvec.append(0)
        while len(colvec)>len(delvec):
            delvec.append(0)
        def cos_similarity(x, y):
            cos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
            return cos
        return cos_similarity(colvec,delvec)

    def getimport(self,poslist,dellist):
        return sum([len(i) for i in poslist])/sum([len(i) for i in dellist])

    def getSGF(self,collist,dellist):
        IC=0
        U = sum([len(i) for i in collist])
        for i in collist:
            ICD=0
            for j in dellist:
                a = set(i)
                b = set(j)
                c = len(list(a & b))/len(i)
                ICD += c*np.log2(c)
            IC-=len(i)*ICD/U
        return IC

    def getICD(self,collist,dellist):
        IC=0
        U = sum([len(i) for i in collist])
        for i in collist:
            ICD=0
            for j in dellist:
                a = set(i)
                b = set(j)
                c = len(list(a & b))/len(i)
                if c!=0:
                    ICD += c*np.log2(c)
            IC-=len(i)*ICD/U
        IRD =0
        for i in collist:
            IRD-= len(i)*np.log2(len(i)/U)/U
        return IRD-IC

    def getmutualinfor(self,poslist,dellist):
        return

    def getcore(self,usedf,featurename=None,decname=None,divlist=None):
        df = usedf
        featurename = featurename or self.df
        if divlist==None:
            decname = decname or self.decision_col
            decdf = df[decname]
            declist = self.divdf(decdf,decname)
        else:
            declist=divlist
        zero_matrix= []
        tmpans = []
        df = df[featurename]
        for i in range(len(df.index.tolist())):
            row=[]
            for a in declist:
                if i in a:
                    flag = a
            for j in range(i,len(df.index.tolist())):
                if j not in flag:
                    row1 = df.iloc[i].tolist()
                    row2 = df.iloc[j].tolist()
                    diff=[]
                    for k in range(len(row1)):
                        if row1[k]!=row2[k]:
                            diff.append(k)
                    diffstr = ''
                    for n in diff:
                        diffstr+=str(n)
                    row.append(diffstr)
                    if len(diffstr)==1 and int(diffstr) not in tmpans:
                        tmpans.append(int(diffstr))
                else:
                    row.append(str(0))
            zero_matrix.append(row)
        return [df.columns.tolist()[index] for index in tmpans]

    def cosfs(self,df,featurecol = None,deccol = None):
        df = df
        featurecol = featurecol or self.feature_col
        decisioncol = deccol or self.decision_col
        divlist = self.divdf(df[decisioncol],decisioncol)
        newfeature = self.getcore(df,featurecol,divlist=divlist)
        tempfeature = list(set(featurecol)-set(newfeature))
        if len(newfeature)==0:
            covnum=0
            cosnum=0
        else:
            tempdiv = self.divdf(df,newfeature)
            temppos= self.getpos(tempdiv,divlist)
            cosnum = self.getcos(tempdiv, divlist)
            covnum = self.getimport(temppos,divlist)
        while covnum!=1.0:
            flag=0
            tempf = -1
            tempdata=[]
            tempcov=0
            for i in tempfeature:
                select_feature = newfeature.copy()
                select_feature.append(i)
                newdiv = self.divdf(df,select_feature)
                newcos = self.getcos(newdiv,divlist)
                newpos = self.getpos(newdiv,divlist)
                newcov = self.getimport(newpos,divlist)
                if newcos>cosnum and newcov>covnum:
                    tempf = i
                    cosnum = newcos
                    tempcov = newcov
                elif newcos==cosnum:
                    if tempcov!=0:
                        if newcov>tempcov:
                            tempf = i
                            cosnum = newcos
                            tempcov = newcov
                    else:
                        if newcov>covnum:
                            tempf = i
                            cosnum = newcos
                            tempcov = newcov
                else:
                    tempdata.append((i,newcos,newcov))
            if tempf==-1:
                tempdata = sorted(tempdata,key=lambda x:x[1],reverse=True)
                for i in tempdata:
                    if i[2]>covnum:
                        newfeature.append(i[0])
                        tempfeature.remove(i[0])
                        flag=1
                        break
                if flag==0:
                    for i in tempdata:
                        if i[2] == covnum:
                            newfeature.append(i[0])
                            tempfeature.remove(i[0])
                            flag = 1
                            break
            else:
                newfeature.append(tempf)
                tempfeature.remove(tempf)
                flag=1
            if flag!=0:
                tempdiv = self.divdf(df, newfeature)
                posnum = self.getpos(tempdiv, divlist)
                cosnum = self.getcos(tempdiv,divlist)
                covnum = self.getimport(posnum, divlist)
            else:
                break
        tempdiv = self.divdf(df, newfeature)
        cosnum = self.getcos(tempdiv, divlist)
        temppos = self.getpos(tempdiv, divlist)
        covnum = self.getimport(temppos, divlist)

        return len(newfeature),cosnum,covnum

    def covfs(self,df,featurecol = None,deccol = None):
        df = df
        featurecol = featurecol or self.feature_col
        decisioncol = deccol or self.decision_col
        divlist = self.divdf(df[decisioncol],decisioncol)
        newfeature = self.getcore(df,featurecol,divlist=divlist)
        tempfeature = list(set(featurecol)-set(newfeature))
        if len(newfeature)==0:
            covnum=0
        else:
            tempdiv = self.divdf(df,newfeature)
            temppos= self.getpos(tempdiv,divlist)
            covnum = self.getimport(temppos,divlist)
        while covnum!=1.0:
            flag=0
            tempf = -1
            if len(tempfeature)!=0:
                for i in tempfeature:
                    select_feature = newfeature.copy()
                    select_feature.append(i)
                    newdiv = self.divdf(df,select_feature)
                    newpos = self.getpos(newdiv,divlist)
                    newcov = self.getimport(newpos,divlist)
                    if newcov>=covnum:
                        tempf = i
                        covnum = newcov
                        flag = 1
                if flag!=0:
                    newfeature.append(tempf)
                    tempfeature.remove(tempf)
                    tempdiv = self.divdf(df, newfeature)
                    posnum = self.getpos(tempdiv, divlist)
                    covnum = self.getimport(posnum, divlist)
                else:
                    break
            else:
                break
        tempdiv = self.divdf(df, newfeature)
        cosnum = self.getcos(tempdiv, divlist)
        temppos = self.getpos(tempdiv, divlist)
        covnum = self.getimport(temppos, divlist)

        return newfeature,len(newfeature),cosnum,covnum

    def SGFfs(self,df,featurecol = None,deccol = None):
        df = df
        featurecol = featurecol or self.feature_col
        decisioncol = deccol or self.decision_col
        divlist = self.divdf(df[decisioncol],decisioncol)
        featurediv = self.divdf(df[featurecol],featurecol)
        covtarget = self.getICD(featurediv,divlist)
        newfeature = self.getcore(df,featurecol,divlist=divlist)
        tempfeature = list(set(featurecol)-set(newfeature))
        if len(newfeature)==0:
            covnum=0
        else:
            tempdiv = self.divdf(df,newfeature)
            covnum = self.getICD(tempdiv,divlist)
        while covnum!=covtarget:
            flag=0
            tempf = -1
            if len(tempfeature)!=0:
                ICDflag =0
                for i in tempfeature:
                    select_feature = newfeature.copy()
                    select_feature.append(i)
                    newdiv = self.divdf(df,select_feature)
                    newcov = self.getICD(newdiv,divlist)-covnum
                    if newcov>ICDflag:
                        tempf = i
                        ICDflag = newcov
                        flag = 1
                if flag!=0:
                    newfeature.append(tempf)
                    tempfeature.remove(tempf)
                    tempdiv = self.divdf(df, newfeature)
                    covnum = self.getICD(tempdiv, divlist)
                else:
                    break
            else:
                break
        tempdiv = self.divdf(df, newfeature)
        cosnum = self.getcos(tempdiv, divlist)
        temppos = self.getpos(tempdiv, divlist)
        covnum = self.getimport(temppos, divlist)

        return len(newfeature),cosnum,covnum
if __name__ == '__main__':
    # df = pd.read_csv('example.csv')
    # RS = cosRoughSet(
    #     data=df,
    #     feature_col=['天气', '事故情形', '事故原因'],
    #     decision_col=['损坏部位']
    # )
    # print(RS.cosfs(df,['天气', '事故情形', '事故原因'],['损坏部位']))
    # print(RS.covfs(df,['天气', '事故情形', '事故原因'],['损坏部位']))
    # data = pd.read_table('./数据/page+blocks+classification/page-blocks.data')
    # print(data)
    data = pd.read_csv('./数据/internet+advertisements.csv')
    print(len(data.columns.tolist()))
    # RS = cosRoughSet(
    #     data=data,
    #     feature_col=list(data.columns[1:len(data.columns.tolist())]),
    #     decision_col=[data.columns[0]]
    # )
    # print(RS.SGFfs(data,list(data.columns[1:len(data.columns.tolist())]),[data.columns[0]]))
    # print(RS.covfs(data,list(data.columns[1:len(data.columns.tolist())]),[data.columns[0]]))
    # print(RS.cosfs(data,list(data.columns[1:len(data.columns.tolist())]),[data.columns[0]]))
    RS = cosRoughSet(
        data=data,
        feature_col=list(data.columns[0:-1]),
        decision_col=[data.columns[-1]]
    )
    print(RS.SGFfs(data, list(data.columns[0:-1]), [data.columns[-1]]))
    print(RS.covfs(data,list(data.columns[0:-1]),[data.columns[-1]]))
    # print(RS.cosfs(data,list(data.columns[0:-1]),[data.columns[-1]]))





