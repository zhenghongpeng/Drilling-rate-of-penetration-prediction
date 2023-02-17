from datetime import datetime, timedelta
from cognite.client.utils import ms_to_datetime
import pickle
import pandas as pd


def handle(client, data=None, secrets=None, function_call_info=None):
    """Handler Function for failure prediction
    Args:
        client : Cognite Client (not needed, it's available to it, when deployed)
        data : data needed by function
        secrets : Any secrets it needs
        function_call_info : any other information about function

    Returns:
        response : response or result from the function

    [requirements]
    pandas
    scikit-learn
    [/requirements]
    """

    #download model 
    file_obj =  client.files.retrieve(external_id="rfc_model_for_cp_wells")
    feature_name_list = file_obj.metadata['feature_list'].split(";")
    # feature list is needed for prediction
    feature_id_list = [int(f) for f in file_obj.metadata['feature_id_list'].split(";")]

    # load the model into memory
    loaded_model = pickle.loads(client.files.download_bytes(id=file_obj.id))
    # load data
    print("input data {}".format(data))
    start_date =  ms_to_datetime(data['start_date'])
    duration = data['days']
    # 30 days of training data chosen arbitrarily
    end_date = start_date + timedelta(days=duration)
    data_for_prediction = client.datapoints.retrieve_dataframe(
        id=feature_id_list,
        start=start_date,
        end=end_date,
        column_names="id"
    )

    feature_id_list_str = file_obj.metadata['feature_id_list'].split(";")
    dict_feature_id_name = dict(zip(feature_id_list_str, feature_name_list))
    # prepare data for prediction based on feature list order
    data_for_prediction = data_for_prediction.rename(columns=dict_feature_id_name)
    # make the response serializable
    predictions = loaded_model.predict(data_for_prediction).tolist()

    # Write the prediction back to corresponding TS asset
    ts_equip_predict_obj = client.time_series.retrieve(external_id = "equipment_failure_prediction")
    dp = {ts_equip_predict_obj.id: predictions}
    idx = data_for_prediction.index
    df_predict = pd.DataFrame(dp, index=idx)
    res=client.datapoints.insert_dataframe(df_predict, external_id_headers=False, dropna=True)
    print("saving prediction {}".format(res))

    return predictions
