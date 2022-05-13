import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind,ttest_rel,f_oneway,zscore
from factor_analyzer import FactorAnalyzer
from BlackBoxAuditing.repairers.GeneralRepairer import Repairer

from util import *
class MLFA:
    """
        Args:
            data: original train set w/ features, target variables, and sensitive attributes
            features: list of features used in model training
            label: name of target variable
            slabel: name of sensitive attribute
            th: threshold to identify the heterogeneous patterns

        """
    def __init__(self, th):
        self.data = np.nan
        self.features = np.nan
        self.label = np.nan
        self.slabel = np.nan
        self.target_features = np.nan
        self.target_fa = np.nan
        self.target_gmm = np.nan
        self.th = th
        self.repairer = np.nan

    def FA(self, df_feature, features):
        ## 1st layer factor analysis
        df_f = df_feature.dropna(subset=features)
        fa = FactorAnalyzer(rotation=None,n_factors=len(features))
        fa.fit(df_f[features])
        ## Check Eigenvalues
        ev, v = fa.get_eigenvalues()
        ## Num. of factors
        nf = len([i for i in ev if i >= 1])
        fa = FactorAnalyzer(rotation=None, n_factors=nf)
        fa.fit(df_f[features])

        df_fa = fa.transform(df_f[features])

        return fa, df_fa

    def cls_samples(self, df_fa):
        X = df_fa
        clist_dic = []
        best_s = 0
        for k in range(2, 15):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=50).fit(X)
            clist = kmeans.labels_
            clist_dic.append(clist)
            s_score = silhouette_score(X, clist)
            # s_list= silhouette_samples(X,clist)
            # print([k, "Silhouette Coefficient: %0.3f"% s_score])
            if (s_score > best_s):
                best_k = k
                best_s = s_score
        # print("best k=%d score: %f"%(best_k,best_s))
        user_group = clist_dic[best_k - 2]
        return list(user_group)

    def cls_features(self, fa, features):
        X = fa.loadings_
        clist_dic = []
        best_s = 0
        for k in range(2, len(features)):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=50).fit(X)
            clist = kmeans.labels_
            clist_dic.append(clist)
            s_score = silhouette_score(X, clist)
            # s_list= silhouette_samples(X,clist)
            # print([k, "Silhouette Coefficient: %0.3f"% s_score])
            if (s_score > best_s):
                best_k = k
                best_s = s_score
        # print("best k=%d score: %f"%(best_k,best_s))
        group_list = clist_dic[best_k - 2]
        group_feature = []
        for g in set(group_list):
            idx = [i for i in range(len(group_list)) if group_list[i] == g]
            fs = np.array(features)[idx]
            # print(g, fs)
            group_feature.append(fs)

        return group_feature

    #### 4. 2nd layer factor analysis
    def FA2(self, df, user_group, group_feature, label, slabel, th=0.5):
        target_features = []  ##features need to rescale
        target_fa = {}
        target_gmm = {}
        target_mean = {}
        target_std = {}

        num_g = []
        for g in set(user_group):
            num_g.append(user_group.count(g))

        max_size = np.max(num_g)
        max_g = np.argmax(num_g)

        for i in range(len(group_feature)):
            features = group_feature[i]
            # print(features)

            dfi = df[list(features) + [label, slabel]].dropna()
            dfi['user_cls'] = user_group
            ## separate each group
            dfs = []
            for g in set(user_group):
                dfg = dfi[dfi['user_cls'] == g]
                if (g != max_g):
                    ## balance each group
                    weights = np.histogram(dfg[features[0]], bins=len(dfg))[0]
                    dff_res = dfg.sample(n=max_size, replace=True, weights=weights, random_state=42)
                    #  mu, sigma=0,0.001
                    #  noise = np.random.normal(mu, sigma, dff_res.shape)
                    dfs.append(dff_res)
                else:
                    dfs.append(dfg)
            df_temp = pd.concat([dfi for dfi in dfs], ignore_index=True)

            ## 2nd layer factor analysis
            # df_temp=(df_temp-df_temp.min())/(df_temp.max()-df_temp.min())

            if (len(features) > 1):
                fa, df_fa = self.FA(df_temp, features)
                df_fa = pd.DataFrame(df_fa)

            else:
                df_fa = df_temp[list(features)]

            falist = df_fa.keys().tolist()
            # df_fa[label]=df_temp[label].tolist()

            #         dfa_temp=df_fa.copy()
            #         dfa_temp[group]=df_temp['user_cls'].tolist()
            #         dfa_temp[label]=df_temp[label].tolist()
            # gmm test
            gmm_null = GaussianMixture(n_components=1, covariance_type='full', max_iter=100).fit(np.array(df_fa))
            bic_null = gmm_null.bic(np.array(df_fa))
            best_bic = bic_null
            dis_null = np.mean(cdist(df_fa, np.array(df_fa.mean()).reshape(1, -1)))
            # print("before ", dis_null)
            for k in range(2, 5):
                gmm_full = GaussianMixture(n_components=k, covariance_type='full', max_iter=100).fit(np.array(df_fa))
                bic_full = gmm_full.bic(df_fa)
                if (bic_full < best_bic):
                    best_bic = bic_full
                    gmm_pred = gmm_full.predict(np.array(df_fa))
                    best_gmm = gmm_full
                else:
                    break
            bic_dis = bic_null - best_bic
            # print(bic_dis, bic_null, best_bic)
            if (bic_dis > 0):
                dfa_temp = df_fa.copy()
                dfa_temp['group'] = gmm_pred  # df_temp['user_cls'].tolist()
                dist = 0
                for g in set(gmm_pred):
                    dfg = dfa_temp[dfa_temp['group'] == g].drop('group', axis=1)
                    dist += np.mean(cdist(dfg, np.array(dfg.mean()).reshape(1, -1)))
                dis_full = dist / len(set(gmm_pred))
                # print("after ", dis_full)
                dfa_temp[label] = df_temp[label].tolist()
                if ((dis_null - dis_full) / dis_null > th):
                    # print("Exist simpson's paradox")
                    # gplot(dfa_temp,falist,label,'group')
                    target_features.append(tuple(features))
                    if (len(features) > 1):
                        target_fa.setdefault(tuple(features), fa)
                    else:
                        target_fa.setdefault(tuple(features), np.nan)
                    target_gmm.setdefault(tuple(features), best_gmm)
                    ##get the convert mean/std
                    df_temp['pred'] = gmm_pred
                    convert_mean = {}
                    convert_std = {}
                    for g in set(gmm_pred):
                        dfg = df_temp[df_temp['pred'] == g][list(features)]
                        convert_mean.setdefault(g, dfg.mean())
                        convert_std.setdefault(g, dfg.std())

                    target_mean.setdefault(tuple(features), convert_mean)
                    target_std.setdefault(tuple(features), convert_std)

        self.target_features = target_features
        self.target_fa = target_fa
        self.target_gmm = target_gmm


    #### remover w/ clustering
    def cls_remover(self, df):
        df_all = []
        t_features = []
        for features in self.target_features:
            t_features += list(features)
            fa = self.target_fa[features]
            gmm = self.target_gmm[features]

            dfi = df[list(features)].copy()
            if (len(features) == 1):
                dfa = dfi
            else:
                dfa = fa.transform(dfi)
            pred = gmm.predict(dfa)
            dfi['group'] = pred

            features1 = dfi.values.tolist()
            index = dfi.columns.get_loc('group')

            repairer = Repairer(features1, index, 1, False)
            repaired_features = repairer.repair(features1)

            dfi_re = pd.DataFrame(np.array(repaired_features), columns=list(features) + ['group'])
            dfi_re['group'] = dfi['group']

            df_all.append(dfi_re.drop('group', axis=1))

        dfn = pd.concat([dff for dff in df_all], axis=1)
        df_norm = df.copy()
        df_norm[t_features] = dfn[t_features].values

        self.repairer = repairer

        return df_norm

    ## converter
    def norm_convert(self, df):
    #def norm_convert(self, df, repairer, target_features, target_fa, target_gmm, target_mean, target_std):
        df_all = []
        t_features = []
        for features in self.target_features:
            t_features += list(features)
            fa = self.target_fa[features]
            gmm = self.target_gmm[features]

            dfi = df[list(features)].copy()
            if (len(features) == 1):
                dfa = dfi
            else:
                dfa = fa.transform(dfi)
            pred = gmm.predict(dfa)

            dfi['group'] = pred

            features1 = dfi.values.tolist()
            index = dfi.columns.get_loc('group')

            #repairer = Repairer(features1, index, 1, False)
            repaired_features = self.repairer.repair(features1)

            dfi_re = pd.DataFrame(np.array(repaired_features), columns=list(features) + ['group'])
            dfi_re['group'] = dfi['group'].tolist()

            df_all.append(dfi_re.drop('group', axis=1))

        dfn = pd.concat([dff for dff in df_all], axis=1)
        df_norm = df.copy()
        df_norm[t_features] = dfn[t_features].values

        return df_norm

    def fit(self, data, features, label, slabel):
        fa, df_fa = self.FA(data, features)
        user_group = self.cls_samples(df_fa)
        group_feature = self.cls_features(fa, features)
        self.FA2(data, user_group, group_feature, label, slabel, self.th)

    def fit_transform(self, data, features, label, slabel):
        self.fit(data, features, label, slabel)
        df_cls_re = self.cls_remover(data)
        return df_cls_re

    def transform(self, df_test):
        if (type(self.target_fa) == float and np.isnan(self.target_fa)):
            raise AssertionError("Please fit the model first by calling function 'fit' or 'fit_transform' !")
        df_test_norm = self.norm_convert(df_test)
        return df_test_norm

if __name__ == "__main__":
    data = pd.read_excel('../cong.xlsx')
    features = ['HEIGHT','WEIGHT','B-CURLS', '6 MInWk','6 STEPS','10 METRE','TIMED UP&GO','5 Sit to Stands',
                'SIT TO STAND','GRIP LEFT','GRIP RIGHT','FR1','FR2','FR3']
    label='TOTAL MISTAKES'
    slabel='SEX '
    mlfa = MLFA(th = 0.1)
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)

    # prediction w/ original data
    y_pred_orig, y_test_orig = model(X_train, X_train[label], X_test, X_test[label], features, label, slabel,
                           model='lr')
    max_g, max_ead = EAD(y_test_orig, y_pred_orig, X_test[slabel], unpriv_group= 2)
    max_g, max_sp = SPD(y_pred_orig, X_test[slabel], unpriv_group=2)
    print(max_g, abs(max_ead), abs(max_sp))

    # prediction w/ data after MLFA
    df_norm_train = mlfa.fit_transform(X_train, features, label, slabel)
    df_norm_test = mlfa.transform(X_test)

    y_pred, y_test = model(df_norm_train, X_train[label], df_norm_test, X_test[label], features, label, slabel, model='lr')
    max_g, max_ead = EAD(y_test, y_pred, df_norm_test[slabel], unpriv_group = 2)
    max_g, max_sp = SPD(y_pred, df_norm_test[slabel], unpriv_group=2)
    print(max_g, abs(max_ead), abs(max_sp))
