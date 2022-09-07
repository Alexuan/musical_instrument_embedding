from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class LabelsToOneHot:
    def __init__(self, data):
        self.labels_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder()

        self.labels_encoder.fit(data)
        data_enc = self.labels_encoder.transform(data).reshape((-1, 1))
        self.one_hot_encoder.fit(data_enc)

    def __call__(self, data):
        data_enc = self.labels_encoder.transform(data).reshape((-1, 1))
        data_onehot = self.one_hot_encoder.transform(data_enc).toarray()
        return data_onehot


class LabelsEncoder:
    def __init__(self, data):
        self.labels_encoder = LabelEncoder()
        self.labels_encoder.fit(data)

    def __call__(self, data):
        return self.labels_encoder.transform(data)


def lable_to_enc_dict(feature_set, onehot=False):
    feature_encoder = LabelsEncoder(feature_set)
    feature_enc = feature_encoder(feature_set)
    if onehot:
        feature_encoder_onehot = LabelsToOneHot(feature_set)
        feature_enc = feature_encoder_onehot(feature_set)

    feature_dict = dict()
    for i in range(len(feature_set)):
        feature_dict[feature_set[i]] = feature_enc[i]
    return feature_dict
