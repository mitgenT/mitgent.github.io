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
        <td>Time,memory</td>
        <td>>=0.20</td>
        <td>API Optimization</td></tr>
 <tr><td>12</td>
        <td>sklearn.linear_model.LogisticRegressionCV</td>
        <td>Time,memory</td>
        <td>>0.21.3</td>
        <td>API Optimization</td></tr>
 <tr><td>13</td>
        <td>sklearn.impute.SimpleImputer</td>
        <td>Time</td>
        <td></td>
        <td>API Optimization</td></tr>
 <tr><td>14</td>
        <td>category_encoders.one_hot.OneHotEncoder</td>
        <td>memory</td>
        <td><=2.1.0</td>
        <td>API Optimization</td></tr>
 <tr><td>15</td>
        <td>lightgbm.LGBMClassifier</td>
        <td>Time,memory</td>
        <td></td>
        <td>API Optimization</td></tr>
 <tr><td>16</td>
        <td>category_encoders.leave_one_out.LeaveOneOutEncoder</td>
        <td>memory</td>
        <td><=2.1.0</td>
        <td>API Optimization</td></tr>
 <tr><td>17</td>
        <td>sklearn.feature_extraction.text.CountVectorizer</td>
        <td>Time</td>
        <td><1.0.1</td>
        <td>API Optimization</td></tr>
<tr><td>18</td>
        <td>catboost.CatBoostClassifier</td>
        <td>Time</td>
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
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>23</td>
        <td>catboost.cv</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>24</td>
        <td>catboost.Pool</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>25</td>
        <td>sklearn.svm.SVR</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>26</td>
        <td>sklearn.linear_model.ElasticNet</td>
        <td>Time</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>27</td>
        <td>sklearn.ensemble.RandomForestRegressor</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>28</td>
        <td>xgboost.sklearn.XGBClassifier</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>29</td>
        <td>xgboost.sklearn.XGBRegressor</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>30</td>
        <td>sklearn.svm.NuSVC</td>
        <td>score,Time</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>31</td>
        <td>sklearn.multioutput.MultiOutputClassifier</td>
        <td>Time</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>32</td>
        <td>sklearn.linear_model.SGDClassifier</td>
        <td>Time,Memory</td>
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
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>37</td>
        <td>sklearn.linear_model.Lasso</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>38</td>
        <td>sklearn.svm.LinearSVC</td>
        <td>Time</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>39</td>
        <td>category_encoders.target_encoder.TargetEncoder</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>40</td>
        <td>sklearn.manifold.TSNE</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>41</td>
        <td>sklearn.ensemble.GradientBoostingRegressor</td>
        <td>Time,Memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>42</td>
        <td>imblearn.over_sampling. RandomOverSampler</td>
        <td>Time</td>
        <td>0.7.1</td>
        <td>API Optimization</td></tr>
<tr><td>43</td>
        <td>mlxtend.classifier.StackingCVClassifier</td>
        <td>Time</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>44</td>
        <td>cv2.imread</td>
        <td>Time</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>45</td>
        <td>optuna.create_study.optimize</td>
        <td>Time</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>46</td>
        <td>sklearn.preprocessing.RobustScaler</td>
        <td>Time</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>47</td>
        <td>tensorflow.keras.layers.BatchNormalization</td>
        <td>Time</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>48</td>
        <td>pandas.DataFrame.divide</td>
        <td>memory</td>
        <td></td>
        <td>API Optimization</td></tr>
<tr><td>
49</td>
<td>pandas,scikit-learn,sklearn.ensemble.GradientBoostingClassifier</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
50</td>
<td>pandas,scikit-learn,sklearn.linear_model.LogisticRegression</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
51</td>
<td>lightgbm,scikit-learn,lightgbm.LGBMRegressor,sklearn.feature_selection.VarianceThreshold</td>
<td>Time</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
52</td>
<td>lightgbm,numpy,lightgbm.train</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
53</td>
<td>lightgbm,pandas,lightgbm.LGBMClassifier</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
54</td>
<td>numpy,scikit-learn,sklearn.feature_extraction.text.TfidfVectorizer</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
55</td>
<td>numpy,scikit-learn,sklearn.linear_model.ElasticNet</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
56</td>
<td>numpy,scikit-learn,sklearn.preprocessing.QuantileTransformer</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
57</td>
<td>lightgbm,lightgbm,scikit-learn,lightgbm.Dataset,lightgbm.train,sklearn.preprocessing.LabelEncoder</td>
<td>Time</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
58</td>
<td>pandas,scikit-learn,sklearn.decomposition.PCA</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
59</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.Input,tensorflow.keras.layers.concatenate,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.layers.Reshape</td>
<td>Time</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
60</td>
<td>numpy,scikit-learn,sklearn.preprocessing.LabelEncoder</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
61</td>
<td>pandas,scikit-learn,sklearn.linear_model.LogisticRegression</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
62</td>
<td>numpy,scikit-learn,sklearn.ensemble.RandomForestRegressor</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
63</td>
<td>numpy,scikit-learn,sklearn.linear_model.Ridge</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
64</td>
<td>pandas,scikit-learn,sklearn.preprocessing.StandardScaler</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
65</td>
<td>pandas,scikit-learn,sklearn.neighbors.KNeighborsClassifier</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
66</td>
<td>transformer,transformer,transformers.DistilBertForSequenceClassification.from_pretrained,transformers.DistilBertTokenizerFast.from_pretrained</td>
<td>Time</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
67</td>
<td>numpy,scikit-learn,sklearn.tree.DecisionTreeClassifier</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
68</td>
<td>numpy,scikit-learn,sklearn.linear_model.Ridge</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
69</td>
<td>pandas,scikit-learn,sklearn.preprocessing.StandardScaler</td>
<td>Time</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
70</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.concatenate,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Embedding,tensorflow.keras.layers.Input,tensorflow.keras.layers.Reshape,tensorflow.keras.layers.SpatialDropout1D,tensorflow.keras.models.Model</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
71</td>
<td>tensorflow,tensorflow,tensorflow.keras.layers.Dense,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
72</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Dense ,tensorflow.keras.optimizers.Adam ,tensorflow.keras.Sequential </td>
<td>unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
73</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.applications.VGG16 ,tensorflow.keras.callbacks.EarlyStopping ,tensorflow.keras.callbacks.ModelCheckpoint ,tensorflow.keras.layers.Dense ,tensorflow.keras.layers.Dropout ,tensorflow.keras.layers.Flatten ,tensorflow.keras.layers.LeakyReLU ,tensorflow.keras.models.Model ,tensorflow.keras.optimizers.Adadelta </td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
74</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
75</td>
<td>tensorflow,tensorflow.keras.layers.Conv2D</td>
<td>unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
76</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
77</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.layers.Dense,tensorflow.keras.losses.SparseCategoricalCrossentropy,tensorflow.keras.optimizers.Adam,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
78</td>
<td>keras,keras,keras,keras,keras,keras.layers.Dense,keras.layers.LSTM,keras.layers.RepeatVector,keras.layers.TimeDistributed,keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
79</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.losses.SparseCategoricalCrossentropy,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
80</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,keras.callbacks.LearningRateScheduler,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.adam,keras.utils.np_utils.to_categorical,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
81</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.LearningRateScheduler,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.optimizers.adam,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
82</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ModelCheckpoint,keras.callbacks.ReduceLROnPlateau,keras.layers.Activation,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.Input,keras.layers.MaxPooling2D,keras.models.Model,keras.optimizers.adam,keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
83</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras.layers.Conv1D,keras.layers.Dense,keras.layers.Embedding,keras.layers.GlobalMaxPooling1D,keras.layers.MaxPool1D,keras.Sequential,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
84</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
85</td>
<td>keras,keras,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,keras.backend.clear_session,keras.utils.to_categorical,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
86</td>
<td>keras,keras,keras,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,keras.callbacks.ReduceLROnPlateau,keras.optimizers.adam,keras.utils.np_utils.to_categorical,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
87</td>
<td>keras,keras,keras,keras,keras,keras,keras.layers.convolutional.Conv2D,keras.layers.convolutional.MaxPooling2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
88</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
89</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.Input,keras.layers.MaxPooling2D,keras.losses.SparseCategoricalCrossentropy,keras.optimizers.adam,keras.Sequential,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
90</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.Input,keras.layers.Maximum,keras.layers.MaxPooling2D,keras.Model,keras.models.Sequential,keras.optimizers.RMSprop,keras.optimizers.schedules.ExponentialDecay,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
91</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
92</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
93</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.models.Sequential,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
94</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,tensorflow,tensorflow,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.LeakyReLU,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.adam,keras.optimizers.RMSprop,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
95</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
96</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.AveragePooling2D,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.ReLU,tensorflow.keras.models.Sequential,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
97</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.AveragePooling2D,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Flatten,keras.models.Sequential,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
98</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
99</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.layers.Activation,keras.layers.AveragePooling2D,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.Input,keras.layers.MaxPool2D,keras.models.Model,keras.optimizers.adam,keras.optimizers.SGD,keras.preprocessing.image.ImageDataGenerator,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
100</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.core.Dense,keras.layers.core.Dropout,keras.layers.Flatten,keras.models.Sequential,keras.optimizers.adam,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
101</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
102</td>
<td>keras,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.SGD</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
103</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.backend.eval,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.losses.CategoricalCrossentropy,tensorflow.keras.optimizers.Adam,tensorflow.keras.optimizers.schedules.ExponentialDecay,tensorflow.keras.regularizers.l2,tensorflow.keras.Sequential,tensorflow.one_hot</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
104</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.layers.PReLU,tensorflow.keras.metrics.Precision,tensorflow.keras.metrics.Recall,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
105</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.applications.DenseNet121,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ModelCheckpoint,tensorflow.keras.layers.Activation,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.Input,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
106</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.cast,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.AlphaDropout,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical,tensorflow.random.set_seed</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
107</td>
<td>keras,keras,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,keras.engine.input_layer.Input,keras.utils.to_categorical,tensorflow.keras.Input,tensorflow.keras.layers.Activation,tensorflow.keras.layers.Average,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.concatenate,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.Model</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
108</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.ELU,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
109</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.LeakyReLU,keras.layers.MaxPool2D,keras.models.Sequential,keras.optimizers.adam</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
110</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.backend.get_value,tensorflow.keras.backend.set_value,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
111</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.models.Sequential,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
112</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
113</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.Activation,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
114</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
115</td>
<td>keras,keras,keras,keras,keras,keras,keras.layers.Activation,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Flatten,keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
116</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.callbacks.TensorBoard,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
117</td>
<td>keras,keras,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,keras.callbacks.ReduceLROnPlateau,keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.layers.Reshape,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.Sequential,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
118</td>
<td>tensorflow,tensorflow,tensorflow.keras.layers.Dense,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
119</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.layers.convolutional.Convolution2D,keras.layers.convolutional.MaxPooling2D,keras.layers.core.Dense,keras.layers.core.Dropout,keras.layers.core.Flatten,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
120</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,tensorflow,tensorflow,keras.backend.get_value,keras.backend.set_value,keras.callbacks.LambdaCallback,keras.callbacks.LearningRateScheduler,keras.callbacks.ModelCheckpoint,keras.initializers.he_normal,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.GlobalAveragePooling2D,keras.layers.Input,keras.layers.Lambda,keras.layers.MaxPooling2D,keras.layers.SpatialDropout2D,keras.models.Model,keras.preprocessing.image.ImageDataGenerator,tensorflow.config.threading.set_inter_op_parallelism_threads,tensorflow.config.threading.set_intra_op_parallelism_threads,tensorflow.random.set_seed</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
121</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.LearningRateScheduler,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
122</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Nadam,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical,tensorflow.losses.Huber</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
123</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.Conv2D,keras.layers.core.Activation,keras.layers.core.Dense,keras.layers.core.Dropout,keras.layers.Flatten,keras.layers.Input,keras.layers.MaxPooling2D,keras.layers.normalization.BatchNormalization,keras.models.Model,keras.optimizers.adam,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
124</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,tensorflow,tensorflow,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.LeakyReLU,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.adam,keras.optimizers.RMSprop,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
125</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.convolutional.Conv2D,keras.layers.convolutional.MaxPooling2D,keras.layers.core.Dense,keras.layers.core.Dropout,keras.layers.core.Flatten,keras.layers.LeakyReLU,keras.layers.normalization.BatchNormalization,keras.models.Sequential,keras.optimizers.Adadelta,keras.preprocessing.image.ImageDataGenerator,keras.regularizers.l2,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
126</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.plot_model,tensorflow.keras.utils.to_categorical,tensorflow.test.gpu_device_name</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
127</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.LearningRateScheduler,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
128</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
129</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.LeakyReLU,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
130</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
131</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
132</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical,tensorflow.test.gpu_device_name</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
133</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.LearningRateScheduler,keras.callbacks.ModelCheckpoint,keras.callbacks.ReduceLROnPlateau,keras.layers.Activation,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.Input,keras.layers.LeakyReLU,keras.layers.MaxPooling2D,keras.layers.merge.concatenate,keras.models.load_model,keras.models.Model,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
134</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.LearningRateScheduler,keras.callbacks.ModelCheckpoint,keras.callbacks.ReduceLROnPlateau,keras.layers.Activation,keras.layers.convolutional.Conv2D,keras.layers.convolutional.MaxPooling2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.LeakyReLU,keras.models.load_model,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
135</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
136</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.backend.sigmoid,tensorflow.keras.layers.Activation,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Nadam,tensorflow.keras.utils.get_custom_objects,tensorflow.test.gpu_device_name</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
137</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,tensorflow,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.models.Sequential,keras.optimizers.adam,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical,tensorflow.compat.v1.ConfigProto,tensorflow.compat.v1.InteractiveSession</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
138</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,tensorflow,tensorflow,keras.callbacks.LearningRateScheduler,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.LeakyReLU,keras.layers.MaxPooling2D,keras.layers.normalization.BatchNormalization,keras.models.Sequential,keras.optimizers.RMSprop,tensorflow.ConfigProto,tensorflow.keras.backend.set_session,tensorflow.Session</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
139</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.Activation,keras.layers.Convolution2D,keras.layers.Dense,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.adam,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
140</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ModelCheckpoint,keras.callbacks.ReduceLROnPlateau,keras.layers.Activation,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.Input,keras.layers.MaxPooling2D,keras.models.Model,keras.optimizers.adam,keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
141</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
142</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
143</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.backend.clear_session,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam,tensorflow.keras.optimizers.schedules.InverseTimeDecay,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.regularizers.l2</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
144</td>
<td>keras,keras,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,keras.preprocessing.image.ImageDataGenerator,keras.regularizers.l2,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
145</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Embedding,tensorflow.keras.layers.LSTM,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
146</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.GlobalMaxPooling2D,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.preprocessing.image.load_img</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
147</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.backend.clip,keras.backend.epsilon,keras.backend.round,keras.backend.sum,keras.callbacks.EarlyStopping,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.layers.ZeroPadding2D,keras.models.Sequential,keras.optimizers.adam,keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
148</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.optimizers.adam,keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
149</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ModelCheckpoint,keras.callbacks.ReduceLROnPlateau,keras.callbacks.TensorBoard,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.GlobalAveragePooling2D,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.adam,keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
150</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,keras.applications.vgg16.VGG16,keras.layers.Activation,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,tensorflow.device</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
151</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,tensorflow,keras.applications.vgg16.VGG16,keras.layers.Activation,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,tensorflow.device,tensorflow.python.client.device_lib.list_local_devices</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
152</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
153</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,keras.callbacks.EarlyStopping,keras.callbacks.LearningRateScheduler,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.LeakyReLU,keras.layers.MaxPooling2D,keras.layers.normalization.BatchNormalization,keras.models.Sequential,keras.optimizers.RMSprop,keras.utils.to_categorical,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
154</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
155</td>
<td>keras,keras,keras.layers.Dense,keras.Sequential</td>
<td>unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
156</td>
<td>keras,keras,keras,keras,keras,keras,keras.backend.clip,keras.backend.epsilon,keras.backend.round,keras.backend.sum,keras.layers.MaxPool2D,keras.layers.ZeroPadding2D</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
157</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.Activation,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.models.Sequential,keras.optimizers.adam</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
158</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.applications.VGG16,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ModelCheckpoint,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.models.Model,tensorflow.keras.optimizers.Adadelta,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
159</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.Activation,keras.layers.AveragePooling2D,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
160</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.Input,tensorflow.keras.layers.concatenate,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.layers.Reshape,tensorflow.keras.Model</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
161</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
162</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adamax,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
163</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.convolutional.Conv2D,keras.layers.convolutional.MaxPooling2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.models.Sequential,keras.optimizers.SGD,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
164</td>
<td>keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
165</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.models.Sequential,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
166</td>
<td>keras,keras,keras,keras,keras,keras.applications.resnet50.ResNet50,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
167</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.initializers.constant,tensorflow.initializers.TruncatedNormal,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.Sequential,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
168</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical,tensorflow.test.gpu_device_name</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
169</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.layers.Activation,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
170</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.data.Dataset.from_tensor_slices,tensorflow.keras.backend.clear_session,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.optimizers.Adam,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
171</td>
<td>keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.models.Sequential,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
172</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.plot_model,tensorflow.keras.utils.to_categorical,tensorflow.test.gpu_device_name</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
173</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
174</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
175</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
176</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
177</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.convolutional.Conv2D,keras.layers.convolutional.MaxPooling2D,keras.layers.core.Dense,keras.layers.core.Dropout,keras.layers.core.Flatten,keras.layers.normalization.BatchNormalization,keras.models.Sequential,keras.optimizers.Adadelta,keras.preprocessing.image.ImageDataGenerator,keras.regularizers.l2</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
178</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.layers.Conv2D,keras.layers.convolutional.MaxPooling2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.models.Sequential,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
179</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,tensorflow,tensorflow,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.RMSprop,keras.utils.np_utils.to_categorical,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
180</td>
<td>keras,keras,keras_preprocessing,keras.models.Sequential ,keras.optimizers.adam ,keras_preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
181</td>
<td>scikit-learn,sklearn.preprocessing.StandardScaler</td>
<td>unknow</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
182</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Dense,tensorflow.keras.losses.MeanSquaredLogarithmicError,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
183</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.convolutional.Conv2D ,keras.layers.convolutional.MaxPooling2D ,keras.layers.core.Dense ,keras.layers.core.Dropout ,keras.layers.core.Flatten ,keras.layers.LeakyReLU ,keras.layers.normalization.BatchNormalization ,keras.models.Sequential ,keras.optimizers.Adadelta ,keras.preprocessing.image.ImageDataGenerator ,keras.regularizers.l2</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
184</td>
<td>keras,keras,keras,keras,keras,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,keras.callbacks.EarlyStopping,keras.callbacks.ModelCheckpoint,keras.callbacks.ReduceLROnPlateau,keras.models.load_model,keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.layers.BatchNormalization ,tensorflow.keras.layers.Conv2D ,tensorflow.keras.layers.Dense ,tensorflow.keras.layers.Dropout ,tensorflow.keras.layers.Flatten ,tensorflow.keras.layers.GlobalMaxPool2D ,tensorflow.keras.layers.MaxPooling2D ,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
185</td>
<td>keras,keras,keras,keras,keras,keras,keras,tensorflow,tensorflow,keras.callbacks.EarlyStopping,keras.layers.Bidirectional,keras.layers.Dense,keras.layers.Dropout,keras.layers.Embedding,keras.layers.LSTM,keras.models.Sequential,tensorflow.python.keras.preprocessing.sequence.pad_sequences ,tensorflow.python.keras.preprocessing.text.Tokenizer</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
186</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,tensorflow,keras.callbacks.EarlyStopping,keras.callbacks.LearningRateScheduler,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.LeakyReLU,keras.layers.MaxPooling2D,keras.layers.normalization.BatchNormalization,keras.models.Sequential,keras.optimizers.RMSprop,keras.utils.to_categorical,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
187</td>
<td>scikit-learn,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,sklearn.model_selection.KFold ,tensorflow.keras.layers.Bidirectional ,tensorflow.keras.layers.Dense ,tensorflow.keras.layers.Embedding ,tensorflow.keras.layers.Input ,tensorflow.keras.models.tensorflow.keras.modelsodel ,tensorflow.keras.preprocessing.sequence.pad_sequences ,tensorflow.keras.preprocessing.text.Tokenizer</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
188</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Input,tensorflow.keras.Sequential,tensorflow.random.set_seed</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
189</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Input,tensorflow.keras.metrics.BinaryCrossentropy,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
190</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.AlphaDropout,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Input,tensorflow.keras.losses.BinaryCrossentropy,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
191</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Input,tensorflow.keras.Sequential,tensorflow.optimizers.Adam,tensorflow.random.set_seed</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
192</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.backend.clear_session,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ModelCheckpoint,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv1D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.Input,tensorflow.keras.layers.MaxPool1D,tensorflow.keras.metrics.AUC,tensorflow.keras.models.load_model,tensorflow.keras.optimizers.Adam,tensorflow.keras.Sequential,tensorflow.random.set_seed</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
193</td>
<td>keras,keras,keras.layers.Dense,keras.models.Sequential</td>
<td>unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
194</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Input,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
195</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.backend.clear_session,tensorflow.keras.layers.Activation,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Input,tensorflow.keras.losses.BinaryCrossentropy,tensorflow.keras.models.Model,tensorflow.keras.optimizers.Adam,tensorflow.random.set_seed</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
196</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.convert_to_tensor,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Input,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
197</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.cast,tensorflow.config.list_physical_devices,tensorflow.data.Dataset.from_tensor_slices,tensorflow.data.Dataset.zip,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.Input,tensorflow.keras.layers.concatenate,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.experimental.preprocessing.CategoryEncoding,tensorflow.keras.layers.experimental.preprocessing.Normalization,tensorflow.keras.layers.experimental.preprocessing.StringLookup,tensorflow.keras.losses.BinaryCrossentropy,tensorflow.keras.Model,tensorflow.keras.optimizers.Adam,tensorflow.keras.optimizers.RMSprop,tensorflow.stack</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
198</td>
<td>keras,keras,keras,keras,tensorflow,tensorflow,tensorflow,keras.callbacks.EarlyStopping,keras.callbacks.ModelCheckpoint,keras.callbacks.ReduceLROnPlateau,keras.wrappers.scikit_learn.KerasRegressor,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
199</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.data.Dataset.from_tensor_slices,tensorflow.device,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ModelCheckpoint,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Input,tensorflow.keras.layers.LSTM,tensorflow.keras.metrics.RootMeanSquaredError,tensorflow.keras.Model,tensorflow.keras.optimizers.Adam</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
200</td>
<td>keras,keras,keras.models.Sequential,keras.optimizers.adam</td>
<td>unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
201</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.data.Dataset.from_tensor_slices,tensorflow.io.decode_jpeg,tensorflow.io.read_file,tensorflow.keras.layers.Activation,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.Input,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.Model,tensorflow.keras.optimizers.Adam</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
202</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.GlobalMaxPooling2D,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.preprocessing.image.load_img</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
203</td>
<td>keras,keras,keras.models.Sequential,keras.optimizers.adam</td>
<td>unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
204</td>
<td>keras,keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
205</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.preprocessing.image.img_to_array,keras.preprocessing.image.load_img,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
206</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.cast,tensorflow.data.Dataset.from_tensor_slices,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.layers.Bidirectional,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Embedding,tensorflow.keras.layers.experimental.preprocessing.TextVectorization,tensorflow.keras.layers.LSTM,tensorflow.keras.losses.MeanSquaredError,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
207</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ModelCheckpoint,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.DepthwiseConv2D,keras.layers.Dropout,keras.layers.Flatten,keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
208</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.concatenate,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Embedding,tensorflow.keras.layers.Input,tensorflow.keras.layers.Reshape,tensorflow.keras.models.Model</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
209</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.Input,tensorflow.keras.layers.concatenate,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.layers.Reshape,tensorflow.keras.Model</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
210</td>
<td>keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
211</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Nadam,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical,tensorflow.losses.Huber</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
212</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
213</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
214</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.argmax,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.callbacks.ModelCheckpoint,tensorflow.keras.layers.Activation,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
215</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Dense,tensorflow.keras.layers.experimental.preprocessing.Normalization,tensorflow.keras.layers.Input,tensorflow.keras.optimizers.SGD,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
216</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.EarlyStopping,keras.callbacks.ReduceLROnPlateau,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
217</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
218</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.initializers.constant,tensorflow.initializers.TruncatedNormal,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.callbacks.ReduceLROnPlateau,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.optimizers.RMSprop,tensorflow.keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.Sequential,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
219</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.optimizers.RMSprop,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
220</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.LearningRateScheduler,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.GlobalAveragePooling2D,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.models.Sequential,tensorflow.keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
221</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.ModelCheckpoint,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.LeakyReLU,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.models.load_model,tensorflow.keras.models.Sequential,tensorflow.keras.optimizers.Adam,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
222</td>
<td>keras,keras,keras,keras,keras,keras,keras.layers.core.Dense,keras.layers.embeddings.Embedding,keras.layers.recurrent.LSTM,keras.models.Sequential,keras.optimizers.adam,keras.optimizers.RMSprop</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
223</td>
<td>tensorflow,tensorflow,tensorflow.keras.layers.Dense,tensorflow.keras.models.Sequential</td>
<td>unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
224</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.preprocessing.image.img_to_array,keras.preprocessing.image.load_img,keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
225</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras.layers.Convolution2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
226</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.layers.AlphaDropout,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Input,tensorflow.keras.losses.BinaryCrossentropy,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
227</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.data.Dataset.from_tensor_slices,tensorflow.io.decode_jpeg,tensorflow.io.read_file,tensorflow.keras.layers.Activation,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.Input,tensorflow.keras.layers.MaxPooling2D,tensorflow.keras.Model,tensorflow.keras.optimizers.Adam</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
228</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.layers.BatchNormalization,tensorflow.keras.layers.Conv2D,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Flatten,tensorflow.keras.layers.MaxPool2D,tensorflow.keras.models.Sequential,tensorflow.keras.utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
229</td>
<td>keras,keras,keras.models.Sequential,keras.optimizers.adam</td>
<td>unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
230</td>
<td>keras,keras,keras,keras,keras,keras,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Flatten,keras.layers.MaxPooling2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
231</td>
<td>keras,keras,keras,keras,tensorflow,tensorflow,tensorflow,keras.callbacks.EarlyStopping,keras.callbacks.ModelCheckpoint,keras.callbacks.ReduceLROnPlateau,keras.wrappers.scikit_learn.KerasRegressor,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.Sequential</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
232</td>
<td>tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow,tensorflow.cast,tensorflow.config.list_physical_devices,tensorflow.data.Dataset.from_tensor_slices,tensorflow.data.Dataset.zip,tensorflow.keras.callbacks.EarlyStopping,tensorflow.keras.Input,tensorflow.keras.layers.concatenate,tensorflow.keras.layers.Dense,tensorflow.keras.layers.Dropout,tensorflow.keras.layers.experimental.preprocessing.CategoryEncoding,tensorflow.keras.layers.experimental.preprocessing.Normalization,tensorflow.keras.layers.experimental.preprocessing.StringLookup,tensorflow.keras.losses.BinaryCrossentropy,tensorflow.keras.Model,tensorflow.keras.optimizers.Adam,tensorflow.keras.optimizers.RMSprop,tensorflow.stack</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
233</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.callbacks.ReduceLROnPlateau,keras.callbacks.TensorBoard,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.Flatten,keras.layers.MaxPool2D,keras.models.Sequential,keras.preprocessing.image.ImageDataGenerator,keras.utils.np_utils.to_categorical</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
234</td>
<td>keras,keras,keras,keras,keras,keras,keras,keras,keras,keras,keras.layers.Activation,keras.layers.AveragePooling2D,keras.layers.BatchNormalization,keras.layers.Conv2D,keras.layers.Dense,keras.layers.Dropout,keras.layers.GlobalAveragePooling2D,keras.layers.MaxPooling2D,keras.models.Sequential,keras.optimizers.adam</td>
<td>unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
235</td>
<td>keras,tensorflow,keras.preprocessing.image.ImageDataGenerator,tensorflow.keras.optimizers.SGD</td>
<td>unknow,unknow</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
236</td>
<td>tensorflow,all_test</td>
<td>test</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
237</td>
<td>tensorflow,tensorflow,tensorflow.keras.layers.Dense,tensorflow.keras.Sequential</td>
<td>test,test</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
238</td>
<td>catboost,catboost.CatBoostClassifier</td>
<td>score(AUC)</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
239</td>
<td>scikit-learn,sklearn.preprocessing.StandardScaler</td>
<td>score</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
240</td>
<td>keras,keras,keras,keras,scikit-learn,keras.layers.Dense,keras.layers.Dropout,keras.layers.LSTM,keras.models.Sequential,sklearn.preprocessing.MinMaxScaler</td>
<td>score</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
241</td>
<td>autokeras,scikit-learn,sklearn.model_selection.train_test_split</td>
<td>score</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
242</td>
<td>scikit-learn,torchvision,sklearn.model_selection.train_test_split</td>
<td>score</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
243</td>
<td>opencv,tensorflow</td>
<td>score</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
244</td>
<td>pandas,tensorflow,tensorflow.keras.optimizers.RMSprop</td>
<td>score</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
245</td>
<td>pandas,tensorflow,tensorflow.keras.layers.BatchNormalization</td>
<td>score</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
246</td>
<td>keras,keras.preprocessing.image.img_to_array</td>
<td>memory</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
247</td>
<td>keras,keras.preprocessing.image.load_img</td>
<td>memory</td>
<td>
</td>
<td>
Type B</td>
</tr>
<tr><td>
248</td>
<td>scikit-learn,sklearn.ensemble.HistGradientBoostingClassifier</td>
<td>memory</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
249</td>
<td>scikit-learn,sklearn.model_selection.KFold</td>
<td>memory</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
250</td>
<td>scikit-learn,xgboost,xgboost.XGBClassifier</td>
<td>memory</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
251</td>
<td>optuna,optuna.create_study</td>
<td>Memory</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
252</td>
<td>pandas,scikit-learn,sklearn.multioutput.MultiOutputRegressor</td>
<td>memory</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
253</td>
<td>pandas,scikit-learn,sklearn.linear_model.BayesianRidge</td>
<td>memory</td>
<td>
</td>
<td>
Type A</td>
</tr>
<tr><td>
254</td>
<td>scikit-learn,sklearn.preprocessing.OneHotEncoder</td>
<td>Time</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
255</td>
<td>scikit-learn,sklearn.compose.ColumnTransformer</td>
<td>Time</td>
<td>
</td>
<td>
</td>
</tr>
<tr><td>
256</td>
<td>imblearn,imblearn.over_sampling.SMOTE</td>
<td>Time</td>
<td>
</td>
<td>
</td>
</tr>
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