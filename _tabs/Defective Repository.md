---
icon: fas fa-stream
order: 2
---
>We constructed a repository containing 164 defective APIs and 106 API combinations.
{: .prompt-info }
<div class="table-wrapper">
    <table id="data-table-basic" class="table table-striped">
    <thead>
      <tr>
        <th>ID</th>
        <th>APIs</th>
        <th>Performance</th>
        <th>Versions</th>
        <!-- <th>Forks</th> -->
        <th>Root Causes</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>sklearn.ensemble.HistGradientBoostingClassifier</td>
        <td>memory</td>
        <td>< 0.24.2</td>
        <td>Technical Debt</td>
      </tr>
      <tr>
        <td>2</td>
        <td>sklearn.ensemble.HistGradientBoostingRegressor</td>
        <td>memory</td>
        <td>< 0.24.2</td>
        <td>Technical Debt</td>
      </tr>
       <tr><td>3</td>
        <td>sklearn.neighbors.KNeighborsClassifier</td>
        <td>memory</td>
        <td><0.24.2</td>
        <td>API Optimization</td></tr>
 <tr><td>4</td>
        <td>category_encoders.TargetEncoder</td>
        <td>memory</td>
        <td>>=2.0.0</td>
        <td>API Optimization</td></tr>
 <tr><td>5</td>
        <td>sklearn.svm.SVC</td>
        <td>score</td>
        <td>>=0.22</td>
        <td>API Optimization</td></tr>
 <tr><td>6</td>
        <td>catboost.CatBoostRegressor</td>
        <td>score</td>
        <td>>0.25.1</td>
        <td>API Optimization</td></tr>
 <tr><td>7</td>
        <td>sklearn.neural_network.MLPClassifier</td>
        <td>memory</td>
        <td>>=0.22</td>
        <td>API Optimization</td></tr>
 <tr><td>8</td>
        <td>category_encoders.ordinal.OrdinalEncoder</td>
        <td>memory</td>
        <td><=2.1.0</td>
        <td>API Optimization</td></tr>
 <tr><td>9</td>
        <td>category_encoders.woe.WOEEncoder</td>
        <td>memory</td>
        <td><=2.1.0</td>
        <td>API Optimization</td></tr>
 <tr><td>10</td>
        <td>lightgbm.train</td>
        <td>memory</td>
        <td>>2.2.3</td>
        <td>API Optimization</td></tr>
 <tr><td>11</td>
        <td>sklearn.linear_model.LinearRegression</td>
        <td>runtime,memory</td>
        <td>>=0.20</td>
        <td>API Optimization</td></tr>
 <tr><td>12</td>
        <td>sklearn.linear_model.LogisticRegressionCV</td>
        <td>runtime,memory</td>
        <td>>0.21.3</td>
        <td>API Optimization</td></tr>
 <tr><td>13</td>
        <td>sklearn.impute.SimpleImputer</td>
        <td>runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
 <tr><td>14</td>
        <td>category_encoders.one_hot.OneHotEncoder</td>
        <td>memory</td>
        <td><=2.1.0</td>
        <td>API Optimization</td></tr>
 <tr><td>15</td>
        <td>lightgbm.LGBMClassifier</td>
        <td>runtime,memory</td>
        <td></td>
        <td>API Optimization</td></tr>
 <tr><td>16</td>
        <td>category_encoders.leave_one_out.LeaveOneOutEncoder</td>
        <td>memory</td>
        <td><=2.1.0</td>
        <td>API Optimization</td></tr>
 <tr><td>17</td>
        <td>sklearn.feature_extraction.text.CountVectorizer</td>
        <td>runtime</td>
        <td><1.0.1</td>
        <td>API Optimization</td></tr>
<tr><td>18</td>
        <td>catboost.CatBoostClassifier</td>
        <td>runtime</td>
        <td>>=0.20.2</td>
        <td>API Optimization</td></tr>
<tr><td>19</td>
        <td>sklearn.preprocessing.LabelEncoder</td>
        <td>memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>20</td>
        <td>sklearn.feature_selection.mutual_info_regression</td>
        <td>memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>21</td>
        <td>sklearn.preprocessing.LabelBinarizer</td>
        <td>memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>22</td>
        <td>catboost.train</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>23</td>
        <td>catboost.cv</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>24</td>
        <td>catboost.Pool</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>25</td>
        <td>sklearn.svm.SVR</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>26</td>
        <td>sklearn.linear_model.ElasticNet</td>
        <td>runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>27</td>
        <td>sklearn.ensemble.RandomForestRegressor</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>28</td>
        <td>xgboost.sklearn.XGBClassifier</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>29</td>
        <td>xgboost.sklearn.XGBRegressor</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>30</td>
        <td>sklearn.svm.NuSVC</td>
        <td>score,runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>31</td>
        <td>sklearn.multioutput.MultiOutputClassifier</td>
        <td>runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>32</td>
        <td>sklearn.linear_model.SGDClassifier</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>33</td>
        <td>sklearn.linear_model.Ridge</td>
        <td>memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>34</td>
        <td>sklearn.naive_bayes.MultinomialNB</td>
        <td>memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>35</td>
        <td>tensorflow.keras.layers.Conv2D</td>
        <td>memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>36</td>
        <td>sklearn.decomposition.PCA</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>37</td>
        <td>sklearn.linear_model.Lasso</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>38</td>
        <td>sklearn.svm.LinearSVC</td>
        <td>runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>39</td>
        <td>category_encoders.target_encoder.TargetEncoder</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>40</td>
        <td>sklearn.manifold.TSNE</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>41</td>
        <td>sklearn.ensemble.GradientBoostingRegressor</td>
        <td>unknow</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>42</td>
        <td>imblearn.over_sampling. RandomOverSampler</td>
        <td>runtime</td>
        <td>0.7.1</td>
        <td>API Optimization</td></tr>
<tr><td>43</td>
        <td>mlxtend.classifier.StackingCVClassifier</td>
        <td>runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>44</td>
        <td>cv2.imread</td>
        <td>runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>45</td>
        <td>optuna.create_study.optimize</td>
        <td>runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>46</td>
        <td>sklearn.preprocessing.RobustScaler</td>
        <td>runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>47</td>
        <td>tensorflow.keras.layers.BatchNormalization</td>
        <td>runtime</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>48</td>
        <td>pandas.DataFrame.divide</td>
        <td>memory</td>
        <td></td>
        <td>API Optimization</td></tr>

    </tbody>
  </table>
</div>

<div id="d-help-win" class="d-help-win" >
    <div id="win-title">Help
        <span id="d-help-colse" clss="close_2" class="close_2">
            × 
        </span>
    </div>
    <div id="win-content">
        <!-- 我们提供了xxx数据集。
        1.
        2.
        3.
        4.
        查看详细复现结果：
        动图！ -->
        <img src="/assets/images/ML-Bug_tu.gif">
    </div>
</div>
<script src="/assets/js/jquery-1.12.4.min.js"></script>
    
<script src="/assets/js/jquery.dataTables.min.js"></script>
<script src="/assets/js/data-table-act.js"></script>