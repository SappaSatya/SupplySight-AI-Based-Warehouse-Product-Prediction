import pickle
import pandas as pd

def predict_product_weight():
    # Load trained model
    with open(r"D:\workspace\arvind\supply-chain-product-prediction-manu\models\final_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Define feature names (same order as training)
    feature_names = [
        "num_refill_req_l3m", "transport_issue_l1y", "Competitor_in_mkt", "retail_shop_num",
        "distributor_num", "flood_impacted", "flood_proof", "electric_supply", "dist_from_hub",
        "workers_num", "storage_issue_reported_l3m", "temp_reg_mach", "wh_breakdown_l3m",
        "govt_check_l3m", "Location_type_Urban", "WH_capacity_size_Mid", "WH_capacity_size_Small",
        "zone_North", "zone_South", "zone_West", "WH_regional_zone_Zone 2", "WH_regional_zone_Zone 3",
        "WH_regional_zone_Zone 4", "WH_regional_zone_Zone 5", "WH_regional_zone_Zone 6",
        "wh_owner_type_Rented", "approved_wh_govt_certificate_A+", "approved_wh_govt_certificate_B",
        "approved_wh_govt_certificate_B+", "approved_wh_govt_certificate_C"
    ]

    # Define corresponding values for each feature
    values = [[
        3,         # num_refill_req_l3m
        1.0,       # transport_issue_l1y
        2.0,       # Competitor_in_mkt
        4651.0,    # retail_shop_num
        24,        # distributor_num
        0,         # flood_impacted
        1,         # flood_proof
        1,         # electric_supply
        91,        # dist_from_hub
        29.0,      # workers_num
        0,         # storage_issue_reported_l3m
        0,         # temp_reg_mach
        0,         # wh_breakdown_l3m
        0,         # govt_check_l3m
        1,         # Location_type_Urban
        0,         # WH_capacity_size_Mid
        1,         # WH_capacity_size_Small
        0,         # zone_North
        0,         # zone_South
        1,         # zone_West
        0,         # WH_regional_zone_Zone 2
        0,         # WH_regional_zone_Zone 3
        1,         # WH_regional_zone_Zone 4
        0,         # WH_regional_zone_Zone 5
        0,         # WH_regional_zone_Zone 6
        1,         # wh_owner_type_Rented
        0,         # approved_wh_govt_certificate_A+
        0,         # approved_wh_govt_certificate_B
        0,         # approved_wh_govt_certificate_B+
        1          # approved_wh_govt_certificate_C
    ]]

    # Convert to DataFrame
    new_input_df = pd.DataFrame(values, columns=feature_names)

    # Predict
    predicted_output = model.predict(new_input_df)

    return predicted_output[0]

