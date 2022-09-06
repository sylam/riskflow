# object list defaults
default = {
    'Integer': 0,
    'Float': 0.0,
    'Percent': 0.0,
    'Text': '',
    'Flot': '[{"label":"None", "data":[[0.0,0.0]]}]',
    'Surface': '[[0.0,1.0], [1.0,0.0]]',
    'Space': '{"0.0":[[0.0,0.0],[0.0,0.0]]}',
    'DateList': 'null',
    'CreditSupportList': '[[0,1]]',
    'DatePicker': ''
}

num_format = {
    'float': {'pattern': '0.000'},
    'int': {'pattern': '0.'},
    'percent': {'pattern': '0.00 %'},
    'currency': {'pattern': '0,0.00'}
}

# this whole thing could be stored as a json file . . .
mapping = {
    'Calibration': {
        'fields': {
            'MLE_Parameters': {'widget': 'Container', 'description': 'MLE Parameters',
                               'value': {"Data_Retrieval_Parameters":
                                             {"Start_Date": "", "End_Date": "", "Length": "", "Frequency": "1d",
                                              "Calendar": "", "Business_Days_In_Year": 252,
                                              "Diagnostics_Error_Level": "Info", "Data_Cleaning_Methods": "",
                                              "Horizon": ""},
                                         "Min_Tenor": "3M", "Reversion_Speed_Lower_Bound": 0.1,
                                         "Reversion_Speed_Upper_Bound": 4.0, "Yield_Volatility_Upper_Bound": "",
                                         "Exact_Solution_Optimisation_Parameters": {
                                             "Max_Iterations": 1000, "Fractional_Tolerance": 0.00000001,
                                             "Downhill_Simplex_Scale": 0.005}
                                         },
                               'sub_fields': ['Data_Retrieval_Parameters', 'Min_Tenor', 'Reversion_Speed_Fixed',
                                              'Reversion_Speed_Lower_Bound', 'Reversion_Speed_Upper_Bound',
                                              'Yield_Volatility_Upper_Bound',
                                              'Exact_Solution_Optimisation_Parameters']
                               },
            'Data_Retrieval_Parameters': {'widget': 'Container', 'description': 'Data Retrieval Parameters',
                                          'value': {"Start_Date": "", "End_Date": "", "Length": "", "Frequency": "1d",
                                                    "Calendar": "", "Business_Days_In_Year": 252,
                                                    "Diagnostics_Error_Level": "Info", "Data_Cleaning_Methods": "",
                                                    "Horizon": ""},
                                          'sub_fields': ['Start_Date', 'End_Date', 'Length', 'Frequency', 'Calendar',
                                                         'Business_Days_In_Year', 'Diagnostics_Error_Level',
                                                         'Data_Cleaning_Methods', 'Horizon']},
            'Exact_Solution_Optimisation_Parameters': {'widget': 'Container',
                                                       'description': 'Exact Solution Optimisation Parameters',
                                                       'value': {'Max_Iterations': 1000,
                                                                 'Fractional_Tolerance': 0.00000001,
                                                                 'Downhill_Simplex_Scale': 0.005},
                                                       'sub_fields': ['Max_Iterations', 'Fractional_Tolerance',
                                                                      'Downhill_Simplex_Scale']},
            'Max_Iterations': {'widget': 'Integer', 'description': 'Max Iterations', 'value': 1000},
            'Fractional_Tolerance': {'widget': 'Float', 'description': 'Fractional Tolerance', 'value': 0.00000001},
            'Downhill_Simplex_Scale': {'widget': 'Float', 'description': 'Downhill Simplex Scale', 'value': 0.005},
            'Start_Date': {'widget': 'DatePicker', 'description': 'Start Date', 'value': default['DatePicker']},
            'End_Date': {'widget': 'DatePicker', 'description': 'End Date', 'value': default['DatePicker']},
            'Frequency': {'widget': 'Text', 'description': 'Frequency', 'value': '1d', 'obj': 'Period'},
            'Length': {'widget': 'Text', 'description': 'Length', 'value': ''},
            'Calendar': {'widget': 'Text', 'description': 'Calendar', 'value': ''},
            'Horizon': {'widget': 'Text', 'description': 'Horizon', 'value': ''},
            'Diagnostics_Error_Level': {'widget': 'Dropdown', 'description': 'Diagnostics Error Level', 'value': 'Info',
                                        'values': ['None', 'Info', 'Warning', 'Error']},
            'Calibration_Method': {'widget': 'Dropdown', 'description': 'Calibration Method', 'value': 'MLE',
                                   'values': ['MLE', 'Pre_Computed_Statistics']},
            'Data_Cleaning_Methods': {'widget': 'Text', 'description': 'Data Cleaning Methods', 'value': ''},
            'Business_Days_In_Year': {'widget': 'Integer', 'description': 'Business Days In Year', 'value': 252},
            'Min_Tenor': {'widget': 'Text', 'description': 'Min Tenor', 'value': '3M', 'obj': 'Period'},
            'Reversion_Speed_Fixed': {'widget': 'Text', 'description': 'Reversion Speed Fixed', 'value': ''},
            'Reversion_Speed_Lower_Bound': {'widget': 'Float', 'description': 'Reversion Speed Lower Bound',
                                            'value': 0.1},
            'Reversion_Speed_Upper_Bound': {'widget': 'Float', 'description': 'Reversion Speed Upper Bound',
                                            'value': 3.0},
            'Yield_Volatility_Upper_Bound': {'widget': 'Text', 'description': 'Yield Volatility Upper Bound',
                                             'value': ''},
            'Number_PCA_Factors': {'widget': 'Integer', 'description': 'Number Of PCA Factors', 'value': 3},
            'Distribution_Type': {'widget': 'Dropdown', 'description': 'Distribution Type', 'value': 'Lognormal',
                                  'values': ['Lognormal', 'Normal']},
            'Use_Pre_Computed_Statistics': {'widget': 'Dropdown', 'description': 'Use Pre Computed Statistics',
                                            'value': 'No', 'values': ['Yes', 'No']},
            'Matrix_Type': {'widget': 'Dropdown', 'description': 'Matrix Type', 'value': 'Correlation',
                            'values': ['Correlation', 'Covariance']},
            'Rate_Drift_Model': {'widget': 'Dropdown', 'description': 'Rate Drift Model', 'value': 'Drift_To_Forward',
                                 'values': ['Drift_To_Forward', 'Drift_To_Blend']}
        },
        'types': {
            'PCAInterestRateModel': ['Calibration_Method', 'Number_PCA_Factors', 'Distribution_Type', 'Matrix_Type',
                                     'Rate_Drift_Model', 'MLE_Parameters'],
            'GBMAssetPriceModel': ['Use_Pre_Computed_Statistics', 'Data_Retrieval_Parameters']
        }
    },

    'Calculation': {

        'fields': {
            'Base_Date': {'widget': 'DatePicker', 'description': 'Base Date', 'value': default['DatePicker']},
            'Calculate': {'widget': 'Dropdown', 'description': 'Calculate', 'value': 'No', 'values': ['Yes', 'No']},
            'Counterparty': {'widget': 'Text', 'description': 'Counterparty', 'value': ''},
            'Collateral_Curve': {'widget': 'Text', 'description': 'Collateral Curve', 'value': ''},
            'Funding_Curve': {'widget': 'Text', 'description': 'Funding Curve', 'value': ''},
            'Collateral_Spread': {'widget': 'Integer', 'description': 'Collateral Spread', 'value': 0},
            'Funding_Spread': {'widget': 'Integer', 'description': 'Funding Spread', 'value': 0},
            'Bank': {'widget': 'Text', 'description': 'Bank', 'value': ''},
            'Deflate_Stochastically': {'widget': 'Dropdown', 'description': 'Deflate Stochastically', 'value': 'Yes',
                                       'values': ['Yes', 'No']},
            'Stochastic_Hazard_Rates': {'widget': 'Dropdown', 'description': 'Stochastic Hazard Rates', 'value': 'No',
                                        'values': ['Yes', 'No']},
            'Gradient': {'widget': 'Dropdown', 'description': 'Gradient', 'value': 'No', 'values': ['Yes', 'No']},
            'Base_time_grid': {'widget': 'Text', 'description': 'Base time grid',
                               'value': '0d 2d 1w(1w) 3m(1m) 2y(3m)'},
            'Dynamic_Scenario_Dates': {'widget': 'Dropdown', 'description': 'Dynamic Scenario Dates',
                                       'value': 'No', 'values': ['Yes', 'No']},
            'Currency': {'widget': 'Text', 'description': 'Currency', 'value': 'ZAR'},
            'Simulation_Batches': {'widget': 'Integer', 'description': 'Simulation Batches', 'value': 1},
            'Batch_Size': {'widget': 'Integer', 'description': 'Batch Size', 'value': 1024},
            'Random_Seed': {'widget': 'Integer', 'description': 'Random Seed', 'value': 5120},
            'Calc_Scenarios': {'widget': 'Dropdown', 'description': 'Calc Scenarios', 'value': 'No',
                               'values': ['Yes', 'No']},
            'Deflation_Interest_Rate': {'widget': 'Text', 'description': 'Deflation Interest Rate',
                                        'value': 'ZAR-SWAP'},
            'Credit_Valuation_Adjustment': {'widget': 'Container', 'description': 'Credit Valuation Adjustment',
                                            'value': {"Calculate": "No", "Counterparty": "", "Bank": "",
                                                      "Deflate_Stochastically": "Yes", "Stochastic_Hazard_Rates": "No",
                                                      "Gradient": "No"},
                                            'sub_fields': ['Calculate', 'Counterparty', 'Bank',
                                                           'Deflate_Stochastically', 'Stochastic_Hazard_Rates',
                                                           'Gradient']},
            'Collateral_Valuation_Adjustment': {'widget': 'Container', 'description': 'Collateral Valuation Adjustment',
                                                'value': {"Calculate": "No", "Collateral_Curve": "",
                                                          "Funding_Curve": "", "Collateral_Spread": 0,
                                                          "Funding_Spread": 0, "Gradient": "No"},
                                                'sub_fields': ['Calculate', 'Collateral_Curve',
                                                               'Funding_Curve', 'Collateral_Spread',
                                                               'Funding_Spread', 'Gradient']},
            'Generate_Cashflows': {'widget': 'Dropdown', 'description': 'Generate Cashflows', 'value': 'Yes',
                                   'values': ['Yes', 'No'], 'Output': 'Cashflows'}
        },
        'types': {
            'CreditMonteCarlo': ['Base_Date', 'Currency', 'Base_time_grid', 'Deflation_Interest_Rate',
                                 'Simulation_Batches', 'Batch_Size', 'Random_Seed', 'Calc_Scenarios',
                                 'Generate_Cashflows', 'Credit_Valuation_Adjustment',
                                 'Collateral_Valuation_Adjustment'],
            'BaseValuation': ['Base_Date', 'Currency']
        }
    },
    'System': {
        'fields': {
            'Base_Currency': {'widget': 'Text', 'description': 'Base Currency', 'value': ''},
            'Description': {'widget': 'Text', 'description': 'Description', 'value': ''},
            'Base_Date': {'widget': 'DatePicker', 'description': 'Base Date', 'value': default['DatePicker']},
            'Exclude_Deals_With_Missing_Market_Data': {'widget': 'Dropdown',
                                                       'description': 'Exclude Deals With Missing Market Data',
                                                       'value': 'Yes', 'values': ['Yes', 'No']},
            'Proxying_Rules_File': {'widget': 'Text', 'description': 'Proxying Rules File', 'value': ''},
            'Script_Base_Scenario_Multiplier': {'widget': 'Float', 'description': 'Script Base Scenario Multiplier',
                                                'value': 1},
            'Correlations_Healing_Method': {'widget': 'Dropdown', 'description': 'Correlations Healing Method',
                                            'value': 'Eigenvalue_Raising',
                                            'values': ['Eigenvalue_Raising', 'Hope_and_Pray']},
            'Grouping_File': {'widget': 'Text', 'description': 'Grouping File', 'value': ''}
        },
        'types': {
            'Config':
                ['Base_Currency', 'Description', 'Base_Date', 'Exclude_Deals_With_Missing_Market_Data',
                 'Proxying_Rules_File', 'Script_Base_Scenario_Multiplier', 'Correlations_Healing_Method',
                 'Grouping_File']
        }
    },
    'Factor': {
        # All supported risk factors - need to append this once new risk factors are developed..
        'types': {
            "Correlation":
                ["Value", "Property_Aliases"],
            "CommodityPrice":
                ["Spot", "Currency", "Interest_Rate", "Property_Aliases"],
            "CommodityPriceVol":
                ["Currency", "Surface", "Property_Aliases"],
            "ConvenienceYield":
                ["Curve", "Currency", "Property_Aliases"],
            "EquityPriceVol":
                ["Surface", "ATM_Vol", "Smile", "Property_Aliases"],
            "FuturesPrice":
                ["Price", "Property_Aliases"],
            "InterestYieldVol":
                ["Space", "Property_Aliases"],
            "InflationRate":
                ["Price_Index", "Seasonal_Adjustment", "Reference_Name", "Day_Count", "Accrual_Calendar", "Currency",
                 "Curve", "Property_Aliases"],
            "FXVol":
                ["Surface", "ATM_Vol", "Smile", "Property_Aliases"],
            "EquityPrice":
                ["Issuer", "Respect_Default", "Jump_Level", "Currency", "Interest_Rate", "Spot", "Property_Aliases"],
            "FxRate":
                ["Domestic_Currency", "Interest_Rate", "Priority", "Spot", "Property_Aliases"],
            "SurvivalProb":
                ["Recovery_Rate", "Minimum_Recovery_Rate", "Issuer", "Curve", "Property_Aliases"],
            "InterestRate":
                ["Sub_Type", "Floor", "Day_Count", "Accrual_Calendar", "Currency", "Curve", "Property_Aliases"],
            "DiscountRate":
                ["Interest_Rate", "Property_Aliases"],
            "HullWhite2FactorModelParameters":
                ["Quanto_FX_Volatility", "Alpha_1", "Sigma_1", "Quanto_FX_Correlation_1", "Alpha_2", "Sigma_2",
                 "Quanto_FX_Correlation_2", "Correlation", "Property_Aliases"],
            "GBMAssetPriceTSModelParameters":
                ["Quanto_FX_Volatility", "Vol", "Quanto_FX_Correlation", "Property_Aliases"],
            "PriceIndex":
                ["Index", "Next_Publication_Date", "Last_Period_Start", "Publication_Period", "Currency",
                 "Property_Aliases"],
            "ForwardPrice":
                ["Currency", "Curve", "Fixings", "Property_Aliases"],
            "ForwardPriceSample":
                ["Offset", "Holiday_Calendar", "Sampling_Convention", "Property_Aliases"],
            "ReferencePrice":
                ["Fixing_Curve", "ForwardPrice", "Property_Aliases"],
            "ReferenceVol":
                ["ForwardPriceVol", "ReferencePrice", "Property_Aliases"],
            "ForwardPriceVol":
                ["Space", "Property_Aliases"],
            "InterestRateVol":
                ["Space", "Property_Aliases"],
            "DividendRate":
                ["Floor", "Currency", "Curve", "Property_Aliases"]
        },

        # field types for the various risk factors - need to explicitly mention all of them
        'fields': {
            'Accrual_Calendar': {'widget': 'Text', 'description': 'Accrual Calendar', 'value': ''},
            'ATM_Vol': {'widget': 'Text', 'description': 'ATM Vol', 'value': ''},
            'Currency': {'widget': 'Text', 'description': 'Currency', 'value': ''},
            'Curve': {'widget': 'Flot', 'description': 'Curve', 'value': default['Flot']},
            'Fixing_Curve': {'widget': 'Flot', 'description': 'Fixing Curve', 'value': default['Flot']},
            'Day_Count': {'widget': 'Dropdown', 'description': 'Day Count', 'value': 'ACT_365',
                          'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']},
            'Domestic_Currency': {'widget': 'Text', 'description': 'Domestic Currency', 'value': ''},
            'Floor': {'widget': 'Text', 'description': 'Floor', 'value': '<undefined>'},
            'ForwardPrice': {'widget': 'Text', 'description': 'ForwardPrice', 'value': '', 'obj': 'Tuple'},
            'ReferencePrice': {'widget': 'Text', 'description': 'ReferencePrice', 'value': '', 'obj': 'Tuple'},
            'ForwardPriceVol': {'widget': 'Text', 'description': 'ForwardPriceVol', 'value': '', 'obj': 'Tuple'},
            'Holiday_Calendar': {'widget': 'Text', 'description': 'Holiday Calendar', 'value': ''},
            'Sampling_Convention': {'widget': 'Dropdown', 'description': 'Sampling Convention',
                                    'value': 'ForwardPriceSampleDaily',
                                    'values': ['ForwardPriceSampleDaily', 'ForwardPriceSampleBullet']},
            'Offset': {'widget': 'Integer', 'description': 'Offset', 'value': 0},
            'Value': {'widget': 'Float', 'description': 'Value', 'value': 0},
            'Issuer': {'widget': 'Text', 'description': 'Issuer', 'value': ''},
            'Index': {'widget': 'Flot', 'description': 'Index', 'value': default['Flot']},
            'Vol': {'widget': 'Flot', 'description': 'Vol', 'value': default['Flot']},
            'Fixings': {'widget': 'Text', 'description': 'Fixings', 'value': ''},
            'Interest_Rate': {'widget': 'Text', 'description': 'Interest Rate', 'value': '', 'obj': 'Tuple'},
            'Jump_Level': {'widget': 'Float', 'description': 'Jump Level', 'value': 0.0, 'obj': 'Percent'},
            'Last_Period_Start': {'widget': 'DatePicker', 'description': 'Last Period Start',
                                  'value': default['DatePicker']},
            'Correlation': {'widget': 'Float', 'description': 'Correlation', 'value': 0},
            'Quanto_FX_Correlation': {'widget': 'Float', 'description': 'Quanto FX Correlation', 'value': 0},
            'Quanto_FX_Correlation_1': {'widget': 'Float', 'description': 'Quanto FX_Correlation 1', 'value': 0},
            'Quanto_FX_Correlation_2': {'widget': 'Float', 'description': 'Quanto FX Correlation 2', 'value': 0},
            'Alpha_1': {'widget': 'Float', 'description': 'Alpha 1', 'value': 0},
            'Alpha_2': {'widget': 'Float', 'description': 'Alpha 2', 'value': 0},
            'Quanto_FX_Volatility': {'widget': 'Flot', 'description': 'Quanto FX Volatility',
                                     'value': default['Flot']},
            'Sigma_1': {'widget': 'Flot', 'description': 'Sigma 1', 'value': default['Flot']},
            'Sigma_2': {'widget': 'Flot', 'description': 'Sigma 2', 'value': default['Flot']},
            'Minimum_Recovery_Rate': {'widget': 'Text', 'description': 'Minimum Recovery Rate', 'value': '<undefined>'},
            'Next_Publication_Date': {'widget': 'DatePicker', 'description': 'Next Publication Date',
                                      'value': default['DatePicker']},
            'Property_Aliases': {'widget': 'Text', 'description': 'Property Aliases', 'value': ''},
            'Price_Index': {'widget': 'Text', 'description': 'Price Index', 'value': '', 'obj': 'Tuple'},
            'Priority': {'widget': 'Float', 'description': 'Priority', 'value': 3},
            'Publication_Period': {'widget': 'Dropdown', 'description': 'Publication Period', 'value': 'Monthly',
                                   'values': ['Monthly', 'Quarterly']},
            'Reference_Name': {'widget': 'Dropdown', 'description': 'Reference Name',
                               'value': 'IndexReferenceInterpolated3M',
                               'values': ['IndexReferenceInterpolated1M', 'IndexReferenceInterpolated2M',
                                          'IndexReferenceInterpolated3M', 'IndexReferenceInterpolated4M']},
            'Respect_Default': {'widget': 'Dropdown', 'description': 'Respect Default', 'value': 'Yes',
                                'values': ['Yes', 'No']},
            'Recovery_Rate': {'widget': 'BoundedFloat', 'description': 'Recovery Rate', 'value': 0.4, 'min': 0.0,
                              'max': 1.0},
            'Seasonal_Adjustment': {'widget': 'Text', 'description': 'Seasonal Adjustment', 'value': ''},
            'Smile': {'widget': 'Text', 'description': 'Smile', 'value': ''},
            'Spot': {'widget': 'Float', 'description': 'Spot', 'value': 0},
            'Price': {'widget': 'Float', 'description': 'Price', 'value': 0},
            'Surface': {'widget': 'Three', 'description': 'Surface', 'value': default['Surface']},
            'Space': {'widget': 'Three', 'description': 'Surface', 'value': default['Space']},
            'Sub_Type': {'widget': 'Text', 'description': 'Sub Type', 'value': ''}
        }
    },
    'Process': {
        # All supported risk stochastic processes - need to append this once new risk processes are developed..
        'types': {
            "GBMAssetPriceTSModelImplied":
                ["Risk_Premium"],
            "HullWhite2FactorImpliedInterestRateModel":
                ["Lambda_1", "Lambda_2"],
            "GBMAssetPriceModel":
                ["Vol", "Drift"],
            "GBMPriceIndexModel":
                ["Vol", "Drift", "Seasonal_Adjustment"],
            "HWHazardRateModel":
                ["Alpha", "Lambda", "sigma"],
            "PCAInterestRateModel":
                ["Reversion_Speed", "Historical_Yield", "Yield_Volatility", "Eigenvectors", "Rate_Drift_Model",
                 "Princ_Comp_Source", "Distribution_Type"],
            "CSForwardPriceModel":
                ["Alpha", "Drift", "sigma"],
            "HullWhite1FactorInterestRateModel":
                ["Alpha", "Lambda", "Sigma", "Quanto_FX_Correlation", "Quanto_FX_Volatility"]
        },

        # field types for the various risk processes - need to explicitly mention all of them
        'fields': {
            'Vol': {'widget': 'Float', 'description': 'Vol', 'value': 0},
            'Drift': {'widget': 'Float', 'description': 'Drift', 'value': 0},
            'Alpha': {'widget': 'Float', 'description': 'Alpha', 'value': 0},
            'Lambda': {'widget': 'Float', 'description': 'Lambda', 'value': 0},
            'Lambda_1': {'widget': 'Float', 'description': 'Lambda 1', 'value': 0},
            'Lambda_2': {'widget': 'Float', 'description': 'Lambda 2', 'value': 0},
            'sigma': {'widget': 'Float', 'description': 'Sigma', 'value': 0},
            'Risk_Premium': {'widget': 'Flot', 'description': 'Risk Premium', 'value': default['Flot']},
            'Quanto_FX_Correlation': {'widget': 'Float', 'description': 'Quanto_FX_Correlation', 'value': 0},
            'Reversion_Speed': {'widget': 'Float', 'description': 'Reversion Speed', 'value': 0},
            'Historical_Yield': {'widget': 'Flot', 'description': 'Historical Yield', 'value': default['Flot']},
            'Yield_Volatility': {'widget': 'Flot', 'description': 'Yield Volatility', 'value': default['Flot']},
            'Sigma': {'widget': 'Flot', 'description': 'Sigma', 'value': default['Flot']},
            'Rate_Drift_Model': {'widget': 'Dropdown', 'description': 'Rate Drift Model', 'value': 'Drift_To_Forward',
                                 'values': ['Drift_To_Forward', 'Drift_To_Blend']},
            'Princ_Comp_Source': {'widget': 'Dropdown', 'description': 'Princ Comp Source', 'value': 'Correlation',
                                  'values': ['Correlation', 'Covariance']},
            'Distribution_Type': {'widget': 'Dropdown', 'description': 'Distribution Type', 'value': 'Lognormal',
                                  'values': ['Lognormal', 'Normal']},
            'Eigenvectors': {'widget': 'Flot', 'description': 'Eigenvectors',
                             'value': '[{"label":"1", "data":[[0.0,0.0]]},{"label":"2", "data":[[0.0,0.0]]},{"label":"3", "data":[[0.0,0.0]]}]'},
            'Quanto_FX_Volatility': {'widget': 'Flot', 'description': 'Quanto FX Volatility',
                                     'value': default['Flot']},
            'Seasonal_Adjustment': {'widget': 'Text', 'description': 'Seasonal Adjustment', 'value': ''}
        },
    },

    # list mapping risk factors to allowable stochastic processes
    'Process_factor_map': {
        "Correlation": [],
        "CommodityPrice": [],
        "CommodityPriceVol": [],
        "ConvenienceYield": [],
        "EquityPriceVol": [],
        "InterestYieldVol": [],
        "FuturesPrice": [],
        "InflationRate": ["HullWhite1FactorInterestRateModel", "PCAInterestRateModel"],
        "FXVol": [],
        "ForwardPrice": ["CSForwardPriceModel"],
        "ForwardPriceVol": [],
        "ForwardPriceSample": [],
        "ReferencePrice": [],
        "ReferenceVol": [],
        "HullWhite2FactorModelParameters": [],
        # "GBMTSImpliedParameters": [],
        "GBMAssetPriceTSModelParameters": [],
        "EquityPrice": ["GBMAssetPriceModel"],
        "FxRate": ["GBMAssetPriceModel", "GBMAssetPriceTSModelImplied"],
        "SurvivalProb": ["HWHazardRateModel"],
        "InterestRate": ["HullWhite1FactorInterestRateModel", "PCAInterestRateModel"],
        "DiscountRate": [],
        "PriceIndex": ["GBMPriceIndexModel"],
        "InterestRateVol": [],
        "DividendRate": ["HullWhite1FactorInterestRateModel", "PCAInterestRateModel"]
    },
    'MarketPrices': {
        # logical groupings
        'groups': {
            'MarketPrices': (
                'group', ['InterestRatePrices', 'GBMTSModelPrices', 'HullWhite2FactorInterestRateModelPrices']),
            'PointFields': ('default', ['FRADeal', 'SwapInterestDeal', 'DepositDeal']),
        },

        # field groups
        'sections': {
            'InterestRatePrices':
                ['FRADeal', 'SwapInterestDeal', 'DepositDeal'],
            'GBMTSModelPrices':
                [],
            'HullWhite2FactorInterestRateModelPrices':
                []
        },

        # supported types
        'types': {
            "InterestRatePrices":
                ["Property_Aliases", "Currency", "Spot_Offset", "Zero_Rate_Grid", "Discount_Rate"],
            "GBMTSModelPrices":
                ["Property_Aliases", "Asset_Price_Volatility"],
            "HullWhite2FactorInterestRateModelPrices":
                ["Property_Aliases", "Swaption_Volatility", "Generate_Instruments", "Generation_Parameters",
                 "Instrument_Definitions"],
            "quote":
                ["Descriptor", "Use", "Quoted_Market_Value", "DealType", "Quote_Type"]
        },

        'properties': {
            'Locked_Dates': ['Maturity_Date', 'Effective_Date'],
        },

        # instrument fields
        'fields': {
            'Generation_Parameters': {'widget': 'Container', 'description': 'Generation Parameters',
                                      'value': {"Last_Tenor": "9Y", "Floating_Frequency": "6M", "First_Tenor": "1Y",
                                                "Day_Count": "ACT_365", "Last_Maturity": "10Y", "First_Start": "1Y",
                                                "Fixed_Frequency": "6M", "Index_Offset": 0, "Last_Start": "9Y",
                                                "First_Maturity": "10Y"},
                                      'sub_fields': ["Last_Tenor", "Floating_Frequency", "First_Tenor", "Day_Count",
                                                     "Last_Maturity", "First_Start", "Fixed_Frequency", "Index_Offset",
                                                     "Last_Start", "First_Maturity"]},
            'Swaption_Volatility': {'widget': 'Text', 'description': 'Swaption Volatility', 'value': ''},
            'Fixed_Frequency': {'widget': 'Text', 'description': 'Fixed Frequency', 'value': '6M', 'obj': 'Period'},
            'Floating_Frequency': {'widget': 'Text', 'description': 'Floating Frequency', 'value': '6M',
                                   'obj': 'Period'},
            'First_Start': {'widget': 'Text', 'description': 'First Start', 'value': '1Y', 'obj': 'Period'},
            'Last_Start': {'widget': 'Text', 'description': 'Last Start', 'value': '9Y', 'obj': 'Period'},
            'First_Tenor': {'widget': 'Text', 'description': 'First Tenor', 'value': '1Y', 'obj': 'Period'},
            'Last_Tenor': {'widget': 'Text', 'description': 'Last Tenor', 'value': '9Y', 'obj': 'Period'},
            'First_Maturity': {'widget': 'Text', 'description': 'First Maturity', 'value': '10Y', 'obj': 'Period'},
            'Last_Maturity': {'widget': 'Text', 'description': 'Last Maturity', 'value': '10Y', 'obj': 'Period'},
            'Day_Count': {'widget': 'Dropdown', 'description': 'Day Count', 'value': 'ACT_365',
                          'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']},
            'Generate_Instruments': {'widget': 'Dropdown', 'description': 'Generate Instruments', 'value': 'No',
                                     'values': ['Yes', 'No']},
            'Index_Offset': {'widget': 'Integer', 'description': 'Index Offset', 'value': 0},
            'Holiday_Calendar': {'widget': 'Text', 'description': 'Holiday Calendar', 'value': ''},
            'Instrument_Definitions': {'widget': 'Table', 'description': 'Instrument Definitions', 'value': 'null',
                                       'sub_types':
                                           [{},
                                            {'type': 'numeric', 'numericFormat': num_format['currency']},
                                            {},
                                            {'type': 'dropdown',
                                             'source': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360',
                                                        'ACT_ACT_ICMA']},
                                            {},
                                            {},
                                            {},
                                            {'type': 'dropdown', 'source': ['Lognormal', 'Normal']},
                                            {'type': 'numeric', 'numericFormat': num_format['int']},
                                            {'type': 'numeric', 'numericFormat': num_format['percent']}],
                                       'obj':
                                           ['Period', 'Period', 'Period', 'Period', 'Text', 'Integer', 'Text', 'Float',
                                            'Percent', 'Text'],
                                       'col_names':
                                           ['Floating_Frequency', 'Weight', 'Holiday_Calendar', 'Day_Count', 'Start',
                                            'Fixed_Frequency', 'Tenor', 'Market_Volatility_Type', 'Index_Offset',
                                            'Market_Volatility']
                                       },
            'Descriptor': {'widget': 'Text', 'description': 'Descriptor', 'value': ''},
            'Discount_Rate': {'widget': 'Text', 'description': 'Discount Rate', 'value': ''},
            'Currency': {'widget': 'Text', 'description': 'Currency', 'value': ''},
            'Property_Aliases': {'widget': 'Text', 'description': 'Property Aliases', 'value': ''},
            'Asset_Price_Volatility': {'widget': 'Text', 'description': 'Asset Price Volatility', 'value': ''},
            'Spot_Offset': {'widget': 'Integer', 'description': 'Spot Offset', 'value': 2},
            'Zero_Rate_Grid': {'widget': 'Text', 'description': 'Zero Rate Grid',
                               'value': '0d 1d 2d 1w 2w 1m 3m 6m 9m 1y 6m1y 2y 6m2y 3y 6m3y 4y 6m4y 5y 6y 7y 8y 9y 10y 15y 20y 25y'},
            'Points': {'widget': 'Container', 'description': 'Points',
                       'value': {"Use": "Yes", "Deal": "", "Descriptor": "", "Quote_Type": "ATM", "DealType": "",
                                 "Quoted_Market_Value": 0.0},
                       'sub_fields': ['Use', 'Deal', 'Descriptor', 'Quote_Type', 'DealType', 'Quoted_Market_Value']},
            'Quote_Type': {'widget': 'Dropdown', 'description': 'Quote Type', 'value': 'ATM', 'values': ['ATM']},
            'Use': {'widget': 'Dropdown', 'description': 'Use', 'value': 'Yes', 'values': ['Yes', 'No']},
            'DealType': {'widget': 'Dropdown', 'description': 'DealType', 'value': 'DepositDeal',
                         'values': ['DepositDeal', 'FRADeal', 'SwapInterestDeal']},
            'Quoted_Market_Value': {'widget': 'Float', 'description': 'Quoted Market Value', 'value': 0.0}
        }
    },
    'Instrument': {
        # logical groupings
        'groups': {
            'New Structure': ('group', ['NettingCollateralSet', 'StructuredDeal']),
            'New Interest Rate Derivative': (
                'default', ['FixedCashflowDeal', 'CFFixedListDeal', 'CFFixedInterestListDeal',
                            'CFFloatingInterestListDeal', 'DepositDeal', 'CapDeal', 'FRADeal',
                            'FloorDeal', 'SwapBasisDeal', 'SwapInterestDeal', 'SwaptionDeal',
                            'YieldInflationCashflowListDeal']),
            'New FX Derivative': (
                'default', ['FXNonDeliverableForward', 'FXForwardDeal', 'FXOptionDeal', 'SwapCurrencyDeal',
                            'FXDiscreteExplicitAsianOption', 'FXOneTouchOption', 'FXBarrierOption',
                            'MtMCrossCurrencySwapDeal']),
            'New Energy Derivative': ('default', ['FloatingEnergyDeal', 'FixedEnergyDeal', 'EnergySingleOption']),
            'New Equity Derivative': ('default', ['EquitySwapLeg', 'EquityForwardDeal', 'EquityOptionDeal',
                                                  'EquitySwapletListDeal', 'EquityDiscreteExplicitAsianOption']),
            'New Credit Derivative': ('default', ['DealDefaultSwap'])
        },

        # field groups
        'sections': {
            'SwapCurrencyDeal.Fields': ['Effective_Date', 'Principal_Exchange', 'Maturity_Date'],
            'CapDeal.Fields': ['Reset_Type', 'Penultimate_Coupon_Date', 'Index_Day_Count', 'Buy_Sell',
                               'Index_Frequency', 'Payment_Calendars', 'Discount_Rate', 'Forecast_Rate', 'Index_Tenor',
                               'Index_Calendars', 'Known_Rates', 'Maturity_Date', 'Payment_Timing', 'Payment_Offset',
                               'Effective_Date', 'Averaging_Method', 'First_Coupon_Date', 'Accrual_Calendars',
                               'Accrual_Day_Count', 'Index_Publication_Calendars', 'Cap_Rate', 'Payment_Interval',
                               'Reset_Frequency', 'Amortisation', 'Discount_Rate_Volatility', 'Currency',
                               'Forecast_Rate_Volatility', 'Index_Offset', 'Principal'],
            'CashflowListDeal.Fields': ['Repo_Rate', 'Recovery_Rate', 'Description', 'Survival_Probability', 'Buy_Sell',
                                        'Settlement_Date', 'Currency', 'Discount_Rate', 'Investment_Horizon', 'Issuer'],
            'CFFixedInterestListDeal.Fields': ['Fixed_Cashflows', 'Settlement_Style', 'Is_Defaultable',
                                               'Settlement_Amount', 'Calendars', 'Settlement_Amount_Is_Clean',
                                               'Rate_Currency'],
            'FixedCashflowDeal.Fields': ['Currency', 'Discount_Rate', 'Calendars', 'Amount', 'Payment_Date'],
            'YieldInflationCashflowListDeal.Fields': ['Index', 'Real_Yield_Cashflows', 'Calendars', 'Is_Forward_Deal'],
            'CFFloatingInterestListDeal.Fields': ['Discount_Rate_Swaption_Volatility', 'Rate_Adjustment_Method',
                                                  'Settlement_Style', 'Forecast_Rate_Swaption_Volatility',
                                                  'Is_Defaultable', 'Settlement_Amount', 'Float_Cashflows',
                                                  'Forecast_Rate_Cap_Volatility', 'Settlement_Amount_Is_Clean',
                                                  'Discount_Rate_Cap_Volatility', 'Rate_Calendars', 'Forecast_Rate',
                                                  'Rate_Sticky_Month_End', 'Accrual_Calendars', 'Rate_Offset'],
            'FXNonDeliverableForward.Fields': ['Sell_Currency', 'Sell_Amount', 'Settlement_Date', 'Settlement_Currency',
                                               'Buy_Amount', 'Discount_Rate', 'Buy_Currency'],
            'FXForwardDeal.Fields': ['Sell_Currency', 'Sell_Amount', 'Settlement_Date', 'Buy_Amount',
                                     'Sell_Discount_Rate', 'Buy_Currency', 'Buy_Discount_Rate'],
            'SwapInterestDeal.Pay': ['Pay_Rate_Type', 'Pay_First_Coupon_Date', 'Pay_Timing', 'Pay_Payment_Offset',
                                     'Pay_Interest_Frequency', 'Pay_Penultimate_Coupon_Date', 'Pay_Accrual_Calendars',
                                     'Pay_Frequency', 'Pay_Payment_Calendars', 'Pay_Day_Count'],
            'FRADeal.Fields': ['Use_Known_Rate', 'Known_Rate', 'Payment_Timing', 'Principal', 'Interest_Rate',
                               'Day_Count', 'Calendars', 'Reset_Date', 'Effective_Date', 'Maturity_Date', 'Currency',
                               'Discount_Rate', 'Borrower_Lender', 'FRA_Rate'],
            'SwapCurrencyDeal.Receive': ['Receive_Amortisation', 'Receive_Known_Rates', 'Receive_Index_Offset',
                                         'Receive_First_Coupon_Date', 'Receive_Margin', 'Receive_Currency',
                                         'Receive_Interest_Rate_Volatility', 'Receive_Index_Publication_Calendars',
                                         'Receive_Fixed_Compounding', 'Receive_Rate_Multiplier', 'Receive_Frequency',
                                         'Receive_Accrual_Calendars', 'Receive_Penultimate_Coupon_Date',
                                         'Receive_Compounding_Method', 'Receive_Interest_Frequency',
                                         'Receive_Index_Tenor', 'Receive_Principal', 'Receive_Timing',
                                         'Receive_Day_Count', 'Receive_Discount_Rate_Volatility', 'Receive_Rate_Type',
                                         'Receive_Index_Frequency', 'Receive_Interest_Rate', 'Receive_Index_Day_Count',
                                         'Receive_Index_Calendars', 'Receive_Discount_Rate',
                                         'Receive_Payment_Calendars', 'Receive_Reset_Type', 'Receive_Rate_Constant',
                                         'Receive_Fixed_Rate', 'Receive_Payment_Offset'],
            'EquitySwapLeg.Fields': ['Accrual_Calendars', 'Adjustment_Method', 'Dividend_Timing', 'Equity',
                                     'Equity_Volatility', 'Equity_Known_Prices', 'Effective_Date', 'First_Coupon_Date',
                                     'Known_Dividends', 'Maturity_Date', 'Payment_Calendars', 'Payment_Frequency',
                                     'Payment_Offset', 'Penultimate_Coupon_Date', 'Principal_Fixed_Variable',
                                     'Roll_Direction', 'Units', 'Include_Dividends', 'Buy_Sell', 'Currency',
                                     'Discount_Rate', 'Principal', 'Payoff_Currency', 'Payoff_Type', 'Reset_Calendars',
                                     'Reset_Offset'],
            'EquitySwapletListDeal.Fields': ['Equity', 'Equity_Currency', 'Currency',
                                             'Discount_Rate', 'Buy_Sell', 'Payoff_Type', 'Amount_Type',
                                             'Equity_Cashflows'],
            'EquityForwardDeal.Fields': ['Forward_Price', 'Buy_Sell', 'Payoff_Type', 'Equity_Volatility',
                                         'Maturity_Date', 'Equity', 'Units', 'Currency', 'Discount_Rate',
                                         'Payoff_Currency'],
            'EquityOptionDeal.Fields': ['Settlement_Style', 'Strike_Price', 'Buy_Sell', 'Payoff_Type', 'Option_Type',
                                        'Equity_Volatility', 'Option_Style', 'Expiry_Date', 'Forward_Price_Date',
                                        'Equity', 'Units', 'Option_On_Forward', 'Currency', 'Discount_Rate',
                                        'Payoff_Currency'],
            'EquityDiscreteExplicitAsianOption.Fields': ['Strike_Price', 'Buy_Sell', 'Payoff_Type', 'Option_Type',
                                                         'Equity_Volatility', 'Expiry_Date', 'Equity', 'Units',
                                                         'Currency', 'Discount_Rate', 'Payoff_Currency',
                                                         'Sampling_Data'],
            'FloorDeal.Fields': ['Reset_Type', 'Penultimate_Coupon_Date', 'Index_Day_Count', 'Buy_Sell',
                                 'Index_Frequency', 'Payment_Calendars', 'Discount_Rate', 'Forecast_Rate',
                                 'Index_Tenor', 'Index_Calendars', 'Known_Rates', 'Maturity_Date', 'Payment_Timing',
                                 'Floor_Rate', 'Payment_Offset', 'Effective_Date', 'Averaging_Method',
                                 'First_Coupon_Date', 'Accrual_Calendars', 'Accrual_Day_Count',
                                 'Index_Publication_Calendars', 'Payment_Interval', 'Reset_Frequency', 'Amortisation',
                                 'Discount_Rate_Volatility', 'Currency', 'Forecast_Rate_Volatility', 'Index_Offset',
                                 'Principal'],
            'FXOptionDeal.Fields': ['Underlying_Amount', 'Settlement_Style', 'Strike_Price', 'Underlying_Currency',
                                    'Buy_Sell', 'Option_Type', 'Option_Style', 'Expiry_Date', 'FX_Volatility',
                                    'Forward_Price_Date', 'Discount_Rate', 'Option_On_Forward', 'Currency'],
            'FXDiscreteExplicitAsianOption.Fields': ['Currency', 'Discount_Rate', 'Expiry_Date', 'FX_Volatility',
                                                     'Option_Type', 'Buy_Sell', 'Underlying_Currency', 'Strike_Price',
                                                     'Underlying_Amount', 'Sampling_Data'],
            'FXOneTouchOption.Fields': ['Payoff_Currency', 'Underlying_Currency', 'Buy_Sell', 'Cash_Payoff',
                                        'Barrier_Monitoring_Frequency', 'Barrier_Price', 'Barrier_Type_One',
                                        'Option_Payment_Timing', 'Expiry_Date', 'FX_Volatility', 'Discount_Rate',
                                        'Currency'],
            'FXBarrierOption.Fields': ['Underlying_Amount', 'Barrier_Monitoring_Frequency', 'Payoff_Currency',
                                       'Barrier_Price', 'Cash_Rebate', 'Strike_Price', 'Underlying_Currency',
                                       'Buy_Sell', 'Option_Type', 'Barrier_Type', 'Expiry_Date', 'FX_Volatility',
                                       'Discount_Rate', 'Currency'],
            'SwapCurrencyDeal.Pay': ['Pay_Discount_Rate', 'Pay_Principal', 'Pay_Amortisation', 'Pay_First_Coupon_Date',
                                     'Pay_Index_Day_Count', 'Pay_Reset_Type', 'Pay_Penultimate_Coupon_Date',
                                     'Pay_Day_Count', 'Pay_Index_Publication_Calendars', 'Pay_Interest_Frequency',
                                     'Pay_Compounding_Method', 'Pay_Frequency', 'Pay_Index_Tenor',
                                     'Pay_Rate_Multiplier', 'Pay_Discount_Rate_Volatility', 'Pay_Rate_Type',
                                     'Pay_Known_Rates', 'Pay_Rate_Constant', 'Pay_Interest_Rate_Volatility',
                                     'Pay_Index_Frequency', 'Pay_Fixed_Rate', 'Pay_Currency', 'Pay_Payment_Offset',
                                     'Pay_Fixed_Compounding', 'Pay_Timing', 'Pay_Index_Offset', 'Pay_Interest_Rate',
                                     'Pay_Margin', 'Pay_Accrual_Calendars', 'Pay_Index_Calendars',
                                     'Pay_Payment_Calendars'],
            'DealDefaultSwap.Fields': ['Upfront_Date', 'Upfront', 'Protection_Paid_At_Maturity',
                                       'Accrued_To_End_Period', 'Penultimate_Coupon_Date', 'First_Coupon_Date',
                                       'ISDA_Standard', 'Survival_Probability', 'Pay_Rate', 'Pay_Frequency',
                                       'Recovery_Rate', 'Name', 'Buy_Sell', 'Amortisation', 'Calendars',
                                       'Accrual_Day_Count', 'Currency', 'Is_Digital', 'Discount_Rate', 'Effective_Date',
                                       'Maturity_Date', 'Digital_Recovery', 'Accrue_Fee', 'Principal'],
            'CFFixedListDeal.Fields': ['Currency', 'Discount_Rate', 'Buy_Sell', 'Description',
                                       'Fixed_Simple_Cashflows'],
            'SwapBasisDeal.Fields': ['Maturity_Date', 'Principal_Exchange', 'Amortisation', 'Currency', 'Discount_Rate',
                                     'Effective_Date', 'Principal'],
            'SwaptionDeal.Fields': ['Floating_Margin', 'Rate_Schedule', 'Reset_Type', 'Settlement_Style',
                                    'Index_Day_Count', 'Swap_Rate', 'Buy_Sell', 'Option_Expiry_Date',
                                    'Forecast_Rate_Volatility', 'Settlement_Date', 'Margin_Schedule', 'Principal',
                                    'Index_Publication_Calendars', 'Swap_Maturity_Date', 'Discount_Rate',
                                    'Forecast_Rate', 'Payer_Receiver', 'Currency', 'Index_Tenor', 'Index_Offset',
                                    'Index_Calendars'],
            'SwapBasisDeal.Pay': ['Pay_First_Coupon_Date', 'Pay_Index_Day_Count', 'Pay_Payment_Offset',
                                  'Pay_Penultimate_Coupon_Date', 'Pay_Rate_Volatility', 'Pay_Day_Count',
                                  'Pay_Index_Publication_Calendars', 'Pay_Interest_Frequency', 'Pay_Compounding_Method',
                                  'Pay_Frequency', 'Pay_Rate_Constant', 'Pay_Rate_Multiplier', 'Pay_Margin',
                                  'Pay_Reset_Type', 'Pay_Known_Rates', 'Pay_Index_Tenor', 'Pay_Index_Frequency',
                                  'Pay_Timing', 'Pay_Index_Offset', 'Pay_Rate', 'Pay_Discount_Rate_Volatility',
                                  'Pay_Accrual_Calendars', 'Pay_Index_Calendars', 'Pay_Payment_Calendars'],
            'Admin': ['Object', 'Reference', 'Tags', 'MtM'],
            'DepositDeal.Fields': ['Currency', 'Discount_Rate', 'Accrual_Calendars', 'Payment_Calendars',
                                   'Accrual_Day_Count', 'First_Coupon_Date', 'Penultimate_Coupon_Date', 'Amortisation',
                                   'Effective_Date', 'Maturity_Date', 'Payment_Frequency', 'Interest_Frequency',
                                   'Payment_Timing', 'Payment_Offset', 'Compounding', 'Rate_Currency',
                                   'FX_Reset_Offset', 'Known_FX_Rates', 'Amount', 'Interest_Rate',
                                   'Interest_Rate_Schedule'],
            'MtMCrossCurrencySwapDeal.Fields': ['Pay_Discount_Rate', 'Pay_Rate_Type', 'Receive_Interest_Rate',
                                                'Maturity_Date', 'Principal_Exchange', 'Principal_Exchange',
                                                'Receive_Discount_Rate', 'Receive_Rate_Type', 'Effective_Date',
                                                'Pay_Currency', 'MtM_Side', 'Receive_Currency'],
            'FloatingEnergyDeal.Fields': ['Currency', 'Discount_Rate', 'Sampling_Type', 'FX_Sampling_Type',
                                          'Average_FX', 'Payer_Receiver', 'Energy_Cashflows', 'Reference_Type',
                                          'Reference_Volatility', 'Payoff_Currency'],
            'FixedEnergyDeal.Fields': ['Currency', 'Discount_Rate', 'Payer_Receiver', 'Energy_Fixed_Cashflows'],
            'EnergySingleOption.Fields': ['Currency', 'Discount_Rate', 'Buy_Sell', 'Sampling_Type', 'FX_Sampling_Type',
                                          'Average_FX', 'Settlement_Date', 'Period_Start', 'Period_End', 'Strike',
                                          'Realized_Average', 'Option_Type', 'FX_Period_Start', 'FX_Period_End',
                                          'FX_Realized_Average', 'Volume', 'Reference_Type', 'Reference_Volatility',
                                          'Payoff_Currency'],
            'SwapInterestDeal.Fields': ['Reset_Type', 'Index_Day_Count', 'Index_Frequency', 'Rate_Multiplier',
                                        'Discount_Rate', 'Index_Tenor', 'Index_Calendars', 'Known_Rates',
                                        'Maturity_Date', 'Interest_Rate', 'Interest_Rate_Volatility', 'Effective_Date',
                                        'Index_Offset', 'Floating_Margin', 'Fixed_Compounding', 'Rate_Constant',
                                        'Compounding_Method', 'Index_Publication_Calendars', 'Amortisation',
                                        'Discount_Rate_Volatility', 'Currency', 'Swap_Rate', 'Principal'],
            'SwaptionDeal.Pay': ['Pay_Amortisation', 'Pay_First_Coupon_Date', 'Pay_Timing', 'Pay_Payment_Offset',
                                 'Pay_Penultimate_Coupon_Date', 'Pay_Payment_Calendars', 'Pay_Frequency',
                                 'Pay_Day_Count', 'Pay_Calendars'],
            'SwapInterestDeal.Receive': ['Receive_Payment_Calendars', 'Receive_Day_Count', 'Receive_Accrual_Calendars',
                                         'Receive_First_Coupon_Date', 'Receive_Penultimate_Coupon_Date',
                                         'Receive_Interest_Frequency', 'Receive_Timing', 'Receive_Frequency',
                                         'Receive_Payment_Offset'],
            'SwaptionDeal.Receive': ['Receive_Penultimate_Coupon_Date', 'Receive_Day_Count',
                                     'Receive_First_Coupon_Date', 'Receive_Amortisation', 'Receive_Payment_Calendars',
                                     'Receive_Timing', 'Receive_Frequency', 'Receive_Calendars',
                                     'Receive_Payment_Offset'],
            'NettingCollateralSet.Fields': ['Agreement_Currency', 'Apply_Closeout_When_Uncollateralized',
                                            'Balance_Currency', 'Opening_Balance', 'Base_Collateral_Call_Date',
                                            'Calendars', 'Collateral_Assets', 'Collateral_Call_Frequency',
                                            'Collateralized', 'Netted', 'Credit_Support_Amounts',
                                            'Funding_Rate', 'Liquidation_Period', 'Settlement_Period'],
            'StructuredDeal.Fields': ['Currency', 'Net_Cashflows', 'Net_Cashflows'],
            'SwapBasisDeal.Receive': ['Receive_Timing', 'Receive_Known_Rates', 'Receive_Index_Offset', 'Receive_Margin',
                                      'Receive_Index_Publication_Calendars', 'Receive_Rate_Multiplier',
                                      'Receive_Frequency', 'Receive_Accrual_Calendars',
                                      'Receive_Penultimate_Coupon_Date', 'Receive_Compounding_Method',
                                      'Receive_Interest_Frequency', 'Receive_Index_Tenor', 'Receive_Rate_Volatility',
                                      'Receive_Discount_Rate_Volatility', 'Receive_Index_Frequency',
                                      'Receive_Index_Day_Count', 'Receive_Day_Count', 'Receive_First_Coupon_Date',
                                      'Receive_Rate', 'Receive_Payment_Calendars', 'Receive_Reset_Type',
                                      'Receive_Rate_Constant', 'Receive_Index_Calendars', 'Receive_Payment_Offset']
        },
        # supported types
        'types': {
            'NettingCollateralSet':
                ['Admin', 'NettingCollateralSet.Fields'],
            'StructuredDeal':
                ['Admin', 'StructuredDeal.Fields'],
            'DepositDeal':
                ['Admin', 'DepositDeal.Fields'],
            'CFFixedInterestListDeal':
                ['Admin', 'CashflowListDeal.Fields', 'CFFixedInterestListDeal.Fields'],
            'CFFixedListDeal':
                ['Admin', 'CFFixedListDeal.Fields'],
            'FixedCashflowDeal':
                ['Admin', 'FixedCashflowDeal.Fields'],
            'MtMCrossCurrencySwapDeal':
                ['Admin', 'MtMCrossCurrencySwapDeal.Fields'],
            'CFFloatingInterestListDeal':
                ['Admin', 'CashflowListDeal.Fields', 'CFFloatingInterestListDeal.Fields'],
            'YieldInflationCashflowListDeal':
                ['Admin', 'CashflowListDeal.Fields', 'YieldInflationCashflowListDeal.Fields'],
            'CapDeal':
                ['Admin', 'CapDeal.Fields'],
            'DealDefaultSwap':
                ['Admin', 'DealDefaultSwap.Fields'],
            'EquitySwapLeg':
                ['Admin', 'EquitySwapLeg.Fields'],
            'EquitySwapletListDeal':
                ['Admin', 'EquitySwapletListDeal.Fields'],
            'EquityForwardDeal':
                ['Admin', 'EquityForwardDeal.Fields'],
            'EquityOptionDeal':
                ['Admin', 'EquityOptionDeal.Fields'],
            'EquityDiscreteExplicitAsianOption':
                ['Admin', 'EquityDiscreteExplicitAsianOption.Fields'],
            'FloatingEnergyDeal':
                ['Admin', 'FloatingEnergyDeal.Fields'],
            'FixedEnergyDeal':
                ['Admin', 'FixedEnergyDeal.Fields'],
            'EnergySingleOption':
                ['Admin', 'EnergySingleOption.Fields'],
            'FRADeal':
                ['Admin', 'FRADeal.Fields'],
            'FXForwardDeal':
                ['Admin', 'FXForwardDeal.Fields'],
            'FXNonDeliverableForward':
                ['Admin', 'FXNonDeliverableForward.Fields'],
            'FXOptionDeal':
                ['Admin', 'FXOptionDeal.Fields'],
            'FXOneTouchOption':
                ['Admin', 'FXOneTouchOption.Fields'],
            'FXBarrierOption':
                ['Admin', 'FXBarrierOption.Fields'],
            'FXDiscreteExplicitAsianOption':
                ['Admin', 'FXDiscreteExplicitAsianOption.Fields'],
            'FloorDeal':
                ['Admin', 'FloorDeal.Fields'],
            'SwapBasisDeal':
                ['Admin', 'SwapBasisDeal.Fields', 'SwapBasisDeal.Pay', 'SwapBasisDeal.Receive'],
            'SwapCurrencyDeal':
                ['Admin', 'SwapCurrencyDeal.Fields', 'SwapCurrencyDeal.Pay', 'SwapCurrencyDeal.Receive'],
            'SwapInterestDeal':
                ['Admin', 'SwapInterestDeal.Fields', 'SwapInterestDeal.Pay', 'SwapInterestDeal.Receive'],
            'SwaptionDeal':
                ['Admin', 'SwaptionDeal.Fields', 'SwaptionDeal.Pay', 'SwaptionDeal.Receive']
        },

        # instrument fields
        'fields': {
            'Netted': {'widget': 'Dropdown', 'description': 'Netted', 'value': 'True', 'values': ['True', 'False']},
            'Collateralized': {'widget': 'Dropdown', 'description': 'Collateralized', 'value': 'False',
                               'values': ['True', 'False']},
            'Use_Known_Rate': {'widget': 'Dropdown', 'description': 'Use Known Rate', 'value': 'No',
                               'values': ['Yes', 'No']},
            'Known_Rate': {'widget': 'Float', 'description': 'Known Rate', 'value': 0, 'obj': 'Percent'},
            'Upfront_Date': {'widget': 'DatePicker', 'description': 'Upfront Date',
                             'value': default['DatePicker']},
            'Upfront': {'widget': 'Float', 'description': 'Upfront', 'value': 0, 'obj': 'Percent'},
            'Apply_Closeout_When_Uncollateralized': {'widget': 'Dropdown',
                                                     'description': 'Apply Closeout When Uncollateralized',
                                                     'value': 'No', 'values': ['Yes', 'No']},
            'Protection_Paid_At_Maturity': {'widget': 'Dropdown', 'description': 'Protection Paid At Maturity',
                                            'value': 'No', 'values': ['Yes', 'No']},
            'Accrued_To_End_Period': {'widget': 'Dropdown', 'description': 'Accrued To End Period', 'value': 'No',
                                      'values': ['Yes', 'No']},
            'ISDA_Standard': {'widget': 'Dropdown', 'description': 'ISDA_Standard', 'value': 'ISDA_03',
                              'values': ['ISDA_03', 'ISDA_09']},
            'Sampling_Type': {'widget': 'Text', 'description': 'Sampling Type', 'value': '', 'obj': 'Tuple'},  # tuple
            'FX_Sampling_Type': {'widget': 'Text', 'description': 'FX Sampling Type', 'value': '', 'obj': 'Tuple'},
            # tuple
            'Equity_Currency': {'widget': 'Text', 'description': 'Equity Currency', 'value': ''},  # tuple
            'Agreement_Currency': {'widget': 'Text', 'description': 'Agreement Currency', 'value': ''},  # tuple
            'Settlement_Currency': {'widget': 'Text', 'description': 'Settlement Currency', 'value': ''},  # tuple
            'Average_FX': {'widget': 'Dropdown', 'description': 'Average FX', 'value': 'No', 'values': ['Yes', 'No']},
            'Include_Dividends': {'widget': 'Dropdown', 'description': 'Include Dividends', 'value': 'Yes',
                                  'values': ['Yes', 'No']},
            'Amount_Type': {'widget': 'Dropdown', 'description': 'Amount_Type', 'value': 'Principal',
                                  'values': ['Principal', 'Shares']},
            'Is_Forward_Deal': {'widget': 'Dropdown', 'description': 'Is Forward Deal', 'value': 'No',
                                'values': ['Yes', 'No']},
            'Principal_Fixed_Variable': {'widget': 'Dropdown', 'description': 'Principal Fixed Variable',
                                         'value': 'Variable', 'values': ['Fixed', 'Variable']},
            'MtM_Side': {'widget': 'Dropdown', 'description': 'MtM Side', 'value': 'Pay', 'values': ['Pay', 'Receive']},
            'Roll_Direction': {'widget': 'Dropdown', 'description': 'Roll Direction', 'value': 'Forward',
                               'values': ['Forward', 'Backward']},
            'Pay_Fixed': {'widget': 'Dropdown', 'description': 'Pay Fixed', 'value': 'Receive',
                          'values': ['Pay', 'Receive']},
            'Reference_Type': {'widget': 'Text', 'description': 'Reference Type', 'value': '', 'obj': 'Tuple'},  # tuple
            'Reference_Volatility': {'widget': 'Text', 'description': 'Reference Volatility', 'value': '',
                                     'obj': 'Tuple'},  # tuple,
            'Survival_Probability': {'widget': 'Text', 'description': 'Survival Probability', 'value': '',
                                     'obj': 'Tuple'},  # tuple,
            'Barrier_Monitoring_Frequency': {'widget': 'Text', 'description': 'Barrier Monitoring Frequency',
                                             'value': '0M', 'obj': 'Period'},
            'Payment_Frequency': {'widget': 'Text', 'description': 'Payment Frequency', 'value': '0M', 'obj': 'Period'},
            'Reset_Calendars': {'widget': 'Text', 'description': 'Reset Calendars', 'value': ''},
            'Cash_Rebate': {'widget': 'Float', 'description': 'Cash Rebate', 'value': 0},
            'Cash_Payoff': {'widget': 'Float', 'description': 'Cash Payoff', 'value': 0},
            'Rate_Offset': {'widget': 'Integer', 'description': 'Rate Offset', 'value': 0},
            'Reset_Offset': {'widget': 'Integer', 'description': 'Reset Offset', 'value': 0},
            'Adjustment_Method': {'widget': 'Dropdown', 'description': 'Rate Adjustment Method', 'value': 'None',
                                  'values': ['None', 'Modified_Following', 'Following', 'Preceding',
                                             'Modified_Preceding']},
            'Rate_Adjustment_Method': {'widget': 'Dropdown', 'description': 'Rate Adjustment Method', 'value': 'None',
                                       'values': ['None', 'Modified_Following', 'Following', 'Preceding',
                                                  'Modified_Preceding']},
            'Rate_Sticky_Month_End': {'widget': 'Dropdown', 'description': 'Rate Sticky Month End', 'value': 'Yes',
                                      'values': ['Yes', 'No']},
            'Dividend_Timing': {'widget': 'Dropdown', 'description': 'Dividend Timing', 'value': 'Terminal',
                                'values': ['Continuous', 'Terminal']},
            'Barrier_Price': {'widget': 'Float', 'description': 'Barrier Price', 'value': 0},
            'Barrier_Type': {'widget': 'Dropdown', 'description': 'Barrier Type', 'value': 'Down_And_In',
                             'values': ['Down_And_In', 'Down_And_Out', 'Up_And_In', 'Up_And_Out']},
            'Barrier_Type_One': {'widget': 'Dropdown', 'description': 'Barrier Type', 'value': 'Up',
                                 'values': ['Up', 'Down']},
            'Option_Payment_Timing': {'widget': 'Dropdown', 'description': 'Payment Timing', 'value': 'Expiry',
                                      'values': ['Touch', 'Expiry']},

            'Equity_Known_Prices': {'widget': 'Table', 'description': 'Equity Known Prices', 'value': 'null',
                                    'sub_types':
                                        [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                         {'type': 'numeric', 'numericFormat': num_format['currency']},
                                         {'type': 'numeric', 'numericFormat': num_format['currency']}
                                         ],
                                    'obj':
                                        'DateEqualList',
                                    'col_names':
                                        ['Date', 'Asset Price', 'FX Rate']
                                    },
            'Known_Dividends': {'widget': 'Table', 'description': 'Known Dividends', 'value': 'null',
                                'sub_types':
                                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                     {'type': 'numeric', 'numericFormat': num_format['currency']}],
                                'obj':
                                    'DateEqualList',
                                'col_names':
                                    ['Date', 'Value']
                                },
            'Sampling_Data': {'widget': 'Table', 'description': 'Sampling_Data', 'value': 'null',
                              'sub_types':
                                  [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                   {'type': 'numeric', 'numericFormat': num_format['currency']},
                                   {'type': 'numeric', 'numericFormat': num_format['currency']}
                                   ],
                              'obj':
                                  ['DatePicker', 'Float', 'Float'],
                              'col_names':
                                  ['Date', 'Price', 'Weight']
                              },
            'Properties': {'widget': 'Table', 'description': 'Properties', 'value': 'null',
                           'sub_types':
                               [{'type': 'numeric', 'numericFormat': num_format['currency']},
                                {'type': 'numeric', 'numericFormat': num_format['currency']},
                                {'type': 'numeric', 'numericFormat': num_format['percent']},
                                {'type': 'numeric', 'numericFormat': num_format['percent']},
                                {'type': 'numeric', 'numericFormat': num_format['currency']},
                                {'type': 'numeric', 'numericFormat': num_format['currency']},
                                {'type': 'numeric', 'numericFormat': num_format['currency']},
                                {'type': 'numeric', 'numericFormat': num_format['percent']},
                                {'type': 'numeric', 'numericFormat': num_format['currency']},
                                {'type': 'numeric', 'numericFormat': num_format['percent']},
                                {'type': 'dropdown', 'source': ['Yes', 'No']},
                                {'type': 'numeric', 'numericFormat': num_format['percent']},
                                {'type': 'dropdown', 'source': ['Yes', 'No']},
                                {'type': 'numeric', 'numericFormat': num_format['percent']},
                                {'type': 'dropdown', 'source': ['Yes', 'No']},
                                {'type': 'numeric', 'numericFormat': num_format['percent']},
                                {'type': 'dropdown', 'source': ['Yes', 'No']},
                                {'type': 'numeric', 'numericFormat': num_format['percent']},
                                {'type': 'numeric', 'numericFormat': num_format['percent']},
                                {'type': 'numeric', 'numericFormat': num_format['currency']},
                                {'type': 'dropdown', 'source': ['Yes', 'No']}
                                ],
                           'obj':
                               ['Float', 'Float', 'Percent', 'Percent', 'Float', 'Float', 'Float', 'Percent', 'Float',
                                'Percent', 'Text', 'Percent', 'Text', 'Percent', 'Text', 'Percent', 'Text', 'Percent',
                                'Percent', 'Float', 'Text'],
                           'col_names':
                               ['First_Cashflow_Index', 'Digital_Payoff', 'Digital_Payoff_Rate', 'Rate_Constant',
                                'Rate_Multiplier', 'Swap_Multiplier', 'Cap_Multiplier', 'Cap_Strike',
                                'Floor_Multiplier', 'Floor_Strike', 'Use_Cap_Lower_Barrier', 'Cap_Lower_Barrier',
                                'Use_Cap_Upper_Barrier', 'Cap_Upper_Barrier', 'Use_Floor_Lower_Barrier',
                                'Floor_Lower_Barrier', 'Use_Floor_Upper_Barrier', 'Floor_Upper_Barrier',
                                'Exponential_Multiplier', 'Exponential_Margin', 'Discounted']
                           },

            'FloatItems': {
                'widget': 'Table', 'description': 'Items', 'value': 'null',
                'sub_types':
                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'dropdown',
                      'source': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']}
                     ],
                'obj':
                    ['DatePicker', 'Float', 'DatePicker', 'DatePicker', 'Text', 'Float',
                     'ResetArray', 'Basis', 'Float', 'DatePicker', 'Float'],
                'col_names':
                    ['Payment_Date', 'Notional', 'Accrual_Start_Date', 'Accrual_End_Date', 'Accrual_Day_Count',
                     'Accrual_Year_Fraction', 'Resets', 'Margin', 'Fixed_Amount', 'FX_Reset_Date', 'Known_FX_Rate']
            },

            'RealYieldItems': {
                'widget': 'Table', 'description': 'Items', 'value': 'null',
                'sub_types':
                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'dropdown',
                      'source': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['percent']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'dropdown', 'source': ['Yes', 'No']}
                     ],
                'obj':
                    ['DatePicker', 'Float', 'DatePicker', 'Float', 'DatePicker', 'Float', 'DatePicker', 'DatePicker',
                     'Text', 'Float', 'Percent', 'Basis', 'Float', 'Text'],
                'col_names':
                    ['Payment_Date', 'Notional', 'Base_Reference_Date', 'Base_Reference_Value', 'Final_Reference_Date',
                     'Final_Reference_Value', 'Accrual_Start_Date', 'Accrual_End_Date', 'Accrual_Day_Count',
                     'Accrual_Year_Fraction', 'Yield', 'Margin', 'Rate_Multiplier', 'Is_Coupon']
            },
            'FixedItems': {
                'widget': 'Table', 'description': 'Items', 'value': 'null',
                'sub_types':
                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['percent']},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'dropdown',
                      'source': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'dropdown', 'source': ['Yes', 'No']},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']}
                     ],
                'obj':
                    ['DatePicker', 'Float', 'Percent', 'DatePicker', 'DatePicker', 'Text', 'Float', 'Float', 'Text',
                     'DatePicker', 'Float'],
                'col_names':
                    ['Payment_Date', 'Notional', 'Rate', 'Accrual_Start_Date', 'Accrual_End_Date', 'Accrual_Day_Count',
                     'Accrual_Year_Fraction', 'Fixed_Amount', 'Discounted', 'FX_Reset_Date', 'Known_FX_Rate']
            },

            'FixedSimpleItems': {
                'widget': 'Table', 'description': 'Items', 'value': 'null',
                'sub_types':
                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']}
                     ],
                'obj':
                    ['DatePicker', 'Float'],
                'col_names':
                    ['Payment_Date', 'Fixed_Amount']
            },

            'EnergyItems': {
                'widget': 'Table', 'description': 'Items', 'value': 'null',
                'sub_types':
                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']}
                     ],
                'obj':
                    ['DatePicker', 'DatePicker', 'DatePicker', 'Float', 'Float', 'Float', 'DatePicker', 'Float',
                     'DatePicker', 'DatePicker', 'Float'],
                'col_names':
                    ['Payment_Date', 'Period_Start', 'Period_End', 'Volume', 'Fixed_Basis', 'Price_Multiplier',
                     'Realized_Average_Date', 'Realized_Average', 'FX_Period_Start', 'FX_Period_End',
                     'FX_Realized_Average']
            },
            'EquityItems': {
                'widget': 'Table', 'description': 'Items', 'value': 'null',
                'sub_types':
                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']}
                     ],
                'obj':
                    ['DatePicker', 'DatePicker', 'DatePicker', 'Float', 'Float', 'Float',
                     'Float', 'Float', 'Float', 'Float', 'Float', 'Float'],
                'col_names':
                    ['Start_Date', 'End_Date', 'Payment_Date', 'Amount', 'Start_Multiplier', 'End_Multiplier',
                     'Dividend_Multiplier', 'Known_Start_Price', 'Known_End_Price', 'Known_Start_FX_Rate',
                     'Known_End_FX_Rate', 'Quanto_FX_Rate']
            },
            'EnergyFixedItems': {
                'widget': 'Table', 'description': 'Items', 'value': 'null',
                'sub_types':
                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                     {'type': 'numeric', 'numericFormat': num_format['currency']},
                     {'type': 'numeric', 'numericFormat': num_format['currency']}
                     ],
                'obj':
                    ['DatePicker', 'Float', 'Float'],
                'col_names':
                    ['Payment_Date', 'Volume', 'Fixed_Price']
            },
            'Funding_Rate': {'widget': 'Text', 'description': 'Funding Rate', 'value': '', 'obj': 'Tuple'},  # tuple
            'Opening_Balance': {'widget': 'Float', 'description': 'Opening Balance', 'value': 0.0},
            'Balance_Currency': {'widget': 'Text', 'description': 'Balance Currency', 'value': ''},  # tuple
            'Settlement_Period': {'widget': 'Integer', 'description': 'Settlement Period', 'value': 0},
            'Liquidation_Period': {'widget': 'Integer', 'description': 'Liquidation Period', 'value': 0},
            'Collateral_Call_Frequency': {'widget': 'Text', 'description': 'Collateral Call Frequency', 'value': '1D',
                                          'obj': 'Period'},
            'Base_Collateral_Call_Date': {'widget': 'DatePicker', 'description': 'Base Collateral Call Date',
                                          'value': default['DatePicker']},
            'Cash_Collateral': {'widget': 'Table', 'description': 'Cash Collateral', 'value': 'null',
                                'sub_types':
                                    [{},
                                     {'type': 'numeric', 'numericFormat': num_format['float']},
                                     {'type': 'numeric', 'numericFormat': num_format['percent']},
                                     {'type': 'numeric', 'numericFormat': num_format['percent']},
                                     {},
                                     {},
                                     {'type': 'numeric', 'numericFormat': num_format['int']}
                                     ],
                                'obj':
                                    ['Text', 'Float', 'Percent', 'Percent', 'Text', 'Text', 'Integer'],
                                'col_names':
                                    ['Currency', 'Amount', 'Haircut_Posted', 'Haircut_Received',
                                     'Collateral_Rate', 'Funding_Rate', 'Liquidation_Period']
                                },
            'Bond_Collateral': {'widget': 'Table', 'description': 'Bond Collateral', 'value': 'null',
                                'sub_types':
                                    [{'type': 'numeric', 'numericFormat': num_format['percent']},
                                     {'type': 'numeric', 'numericFormat': num_format['percent']},
                                     {'type': 'numeric', 'numericFormat': num_format['int']},
                                     {},
                                     {},
                                     {},
                                     {},
                                     {'type': 'numeric', 'numericFormat': num_format['int']},
                                     {'type': 'numeric', 'numericFormat': num_format['percent']},
                                     {},
                                     {},
                                     {},
                                     ],
                                'obj':
                                    ['Percent', 'Percent', 'Integer', 'Text', 'Text', 'Text', 'Period', 'Float',
                                     'Percent', 'Period', 'Text', 'Text'],
                                'col_names':
                                    ['Haircut_Posted', 'Haircut_Received', 'Liquidation_Period', 'Issuer', 'Currency',
                                     'Discount_Rate', 'Maturity', 'Principle', 'Coupon_Rate', 'Coupon_Interval',
                                     'Collateral_Rate', 'Funding_Rate']
                                },
            'Equity_Collateral': {'widget': 'Table', 'description': 'Equity_Collateral', 'value': 'null',
                                  'sub_types':
                                      [{},
                                       {'type': 'numeric', 'numericFormat': num_format['float']},
                                       {'type': 'numeric', 'numericFormat': num_format['percent']},
                                       {'type': 'numeric', 'numericFormat': num_format['percent']},
                                       {},
                                       {},
                                       {'type': 'numeric', 'numericFormat': num_format['int']}
                                       ],
                                  'obj':
                                      ['Text', 'Float', 'Percent', 'Percent', 'Text', 'Text', 'Integer'],
                                  'col_names':
                                      ['Equity', 'Units', 'Haircut_Posted', 'Haircut_Received',
                                       'Collateral_Rate', 'Funding_Rate', 'Liquidation_Period']
                                  },
            'Commodity_Collateral': {'widget': 'Table', 'description': 'Commodity Collateral', 'value': 'null',
                                     'sub_types':
                                         [{},
                                          {'type': 'numeric', 'numericFormat': num_format['float']},
                                          {'type': 'numeric', 'numericFormat': num_format['percent']},
                                          {'type': 'numeric', 'numericFormat': num_format['percent']},
                                          {},
                                          {},
                                          {'type': 'numeric', 'numericFormat': num_format['int']}
                                          ],
                                     'obj':
                                         ['Text', 'Float', 'Percent', 'Percent', 'Text', 'Text', 'Integer'],
                                     'col_names':
                                         ['Commodity', 'Units', 'Haircut_Posted', 'Haircut_Received',
                                          'Collateral_Rate', 'Funding_Rate', 'Liquidation_Period']
                                     },
            'Underlying_Amount': {'widget': 'Float', 'description': 'Underlying Amount', 'value': 0.0},
            'Pay_Amortisation': {'widget': 'Table', 'description': 'Pay Amortisation',
                                 'value': default['DateList'],
                                 'sub_types':
                                     [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                      {'type': 'numeric', 'numericFormat': num_format['currency']}],
                                 'obj':
                                     'DateList',
                                 'col_names':
                                     ['Date', 'Amount']
                                 },
            'Pay_First_Coupon_Date': {'widget': 'DatePicker', 'description': 'Pay First Coupon Date',
                                      'value': default['DatePicker']},
            'Payment_Date': {'widget': 'DatePicker', 'description': 'Payment Date',
                             'value': default['DatePicker']},
            'Index_Frequency': {'widget': 'Text', 'description': 'Index Frequency', 'value': '0M', 'obj': 'Period'},
            'Rate_Currency': {'widget': 'Text', 'description': 'Rate Currency', 'value': ''},  # tuple
            'Pay_Payment_Offset': {'widget': 'Integer', 'description': 'Pay Payment Offset', 'value': 0},
            'Receive_Amortisation': {'widget': 'Table', 'description': 'Receive Amortisation',
                                     'value': default['DateList'],
                                     'sub_types':
                                         [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                          {'type': 'numeric', 'numericFormat': num_format['currency']}],
                                     'obj':
                                         'DateList',
                                     'col_names':
                                         ['Date', 'Amount']
                                     },
            'Buy_Amount': {'widget': 'Float', 'description': 'Buy Amount', 'value': 0.0},
            'Amount': {'widget': 'Float', 'description': 'Amount', 'value': 0.0},
            'Payment_Calendars': {'widget': 'Text', 'description': 'Payment Calendars', 'value': ''},
            'Pay_Rate_Volatility': {'widget': 'Text', 'description': 'Pay Rate Volatility', 'value': '',
                                    'obj': 'Tuple'},  # tuple
            'Discount_Rate_Swaption_Volatility': {'widget': 'Text', 'description': 'Discount Rate Swaption Volatility',
                                                  'value': '', 'obj': 'Tuple'},  # tuple
            'Settlement_Style': {'widget': 'Dropdown', 'description': 'Settlement Style', 'value': 'Physical',
                                 'values': ['Physical', 'Cash']},
            'Reset_Frequency': {'widget': 'Text', 'description': 'Reset Frequency', 'value': '3M', 'obj': 'Period'},
            'Receive_Rate_Multiplier': {'widget': 'Float', 'description': 'Receive Rate Multiplier', 'value': 1.0},
            'Receive_Frequency': {'widget': 'Text', 'description': 'Receive Frequency', 'value': '3M', 'obj': 'Period'},
            'Object': {'widget': 'Text', 'description': 'Object', 'value': ''},
            'Pay_Compounding_Method': {'widget': 'Dropdown', 'description': 'Pay Compounding Method', 'value': 'None',
                                       'values': ['None', 'Include_Margin', 'Flat', 'Exclude_Margin', 'Exponential']},
            'Payment_Timing': {'widget': 'Dropdown', 'description': 'Payment Timing', 'value': 'End',
                               'values': ['End', 'Begin', 'Discounted']},
            'Pay_Frequency': {'widget': 'Text', 'description': 'Pay Frequency', 'value': '3M', 'obj': 'Period'},
            'Pay_Index_Tenor': {'widget': 'Text', 'description': 'Pay Index Tenor', 'value': '0M', 'obj': 'Period'},
            'Effective_Date': {'widget': 'DatePicker', 'description': 'Effective Date',
                               'value': default['DatePicker']},
            'Reset_Date': {'widget': 'DatePicker', 'description': 'Reset Date', 'value': default['DatePicker']},
            'Investment_Horizon': {'widget': 'DatePicker', 'description': 'Investment Horizon',
                                   'value': default['DatePicker']},
            'Pay_Calendars': {'widget': 'Text', 'description': 'Pay Calendars', 'value': ''},
            'Pay_Reset_Type': {'widget': 'Dropdown', 'description': 'Pay Reset Type', 'value': 'Standard',
                               'values': ['Standard', 'Advance', 'Arrears']},
            'Digital_Recovery': {'widget': 'Float', 'description': 'Digital Recovery', 'value': 0.0, 'obj': 'Percent'},
            'Pay_Index_Frequency': {'widget': 'Text', 'description': 'Pay Index Frequency', 'value': '0M',
                                    'obj': 'Period'},
            'Cap_Rate': {'widget': 'Float', 'description': 'Cap Rate', 'value': 0.0},
            'Buy_Currency': {'widget': 'Text', 'description': 'Buy Currency', 'value': ''},  # tuple
            'Receive_Index_Tenor': {'widget': 'Text', 'description': 'Receive Index Tenor', 'value': '0M',
                                    'obj': 'Period'},
            'Period_Start': {'widget': 'DatePicker', 'description': 'Period Start',
                             'value': default['DatePicker']},
            'Period_End': {'widget': 'DatePicker', 'description': 'Period End', 'value': default['DatePicker']},
            'FX_Period_Start': {'widget': 'DatePicker', 'description': 'FX Period Start',
                                'value': default['DatePicker']},
            'FX_Period_End': {'widget': 'DatePicker', 'description': 'FX Period End',
                              'value': default['DatePicker']},
            'Pay_Fixed_Compounding': {'widget': 'Dropdown', 'description': 'Pay Fixed Compounding', 'value': 'No',
                                      'values': ['Yes', 'No']},
            'Compounding': {'widget': 'Dropdown', 'description': 'Compounding', 'value': 'No', 'values': ['Yes', 'No']},
            'Reference': {'widget': 'Text', 'description': 'Reference', 'value': ''},
            'Underlying_Currency': {'widget': 'Text', 'description': 'Underlying Currency', 'value': ''},  # tuple
            'Option_Expiry_Date': {'widget': 'DatePicker', 'description': 'Option Expiry Date',
                                   'value': default['DatePicker']},
            'Amortisation': {'widget': 'Table', 'description': 'Amortisation', 'value': default['DateList'],
                             'sub_types':
                                 [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                  {'type': 'numeric', 'numericFormat': num_format['currency']}],
                             'obj':
                                 'DateList',
                             'col_names':
                                 ['Date', 'Amount']
                             },
            'Known_FX_Rates': {'widget': 'Table', 'description': 'Known FX Rates', 'value': default['DateList'],
                               'sub_types':
                                   [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                    {'type': 'numeric', 'numericFormat': num_format['currency']}],
                               'obj':
                                   'DateList',
                               'col_names':
                                   ['Date', 'Amount']
                               },
            'Interest_Rate_Schedule': {'widget': 'Table', 'description': 'Interest Rate Schedule',
                                       'value': default['DateList'],
                                       'sub_types':
                                           [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                            {'type': 'numeric', 'numericFormat': num_format['currency']}],
                                       'obj':
                                           'DateList',
                                       'col_names':
                                           ['Date', 'Amount']
                                       },
            'Sell_Currency': {'widget': 'Text', 'description': 'Sell Currency', 'value': ''},  # tuple
            'Pay_Interest_Rate': {'widget': 'Text', 'description': 'Pay Interest Rate', 'value': '', 'obj': 'Tuple'},
            # tuple
            'Option_Style': {'widget': 'Dropdown', 'description': 'Option Style', 'value': 'European',
                             'values': ['European', 'American']},
            'Pay_Rate_Multiplier': {'widget': 'Float', 'description': 'Pay Rate Multiplier', 'value': 1.0},
            'Receive_Payment_Offset': {'widget': 'Integer', 'description': 'Receive Payment Offset', 'value': 0},
            'FX_Reset_Offset': {'widget': 'Integer', 'description': 'FX Reset Offset', 'value': 0},
            'Pay_Principal': {'widget': 'Float', 'description': 'Pay Principal', 'value': 0.0},
            'Penultimate_Coupon_Date': {'widget': 'DatePicker', 'description': 'Penultimate Coupon Date',
                                        'value': default['DatePicker']},
            'Pay_Index_Day_Count': {'widget': 'Dropdown', 'description': 'Pay Index Day Count', 'value': 'ACT_365',
                                    'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360',
                                               'ACT_ACT_ICMA']},
            'Averaging_Method': {'widget': 'Dropdown', 'description': 'Averaging Method', 'value': 'Average_Rate',
                                 'values': ['Average_Interest', 'Average_Rate']},
            'Independent_Amount_Reference': {'widget': 'Dropdown', 'description': 'Independent Amount Reference',
                                             'value': 'None', 'values': ['None', 'Bank', 'Counterparty']},
            'Rate_Multiplier': {'widget': 'Float', 'description': 'Rate Multiplier', 'value': 1.0},
            'Sell_Amount': {'widget': 'Float', 'description': 'Sell Amount', 'value': 0.0},
            'Margin_Schedule': {'widget': 'Table', 'description': 'Margin Schedule', 'value': default['DateList'],
                                'sub_types':
                                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                     {'type': 'numeric', 'numericFormat': num_format['currency']}],
                                'obj':
                                    'DateList',
                                'col_names':
                                    ['Date', 'Amount']
                                },
            'Independent_Amount': {'widget': 'Table', 'description': 'Independent Amount',
                                   'value': default['CreditSupportList'],
                                   'sub_types':
                                       [{'type': 'numeric', 'numericFormat': num_format['currency']},
                                        {'type': 'numeric', 'numericFormat': num_format['int']}],
                                   'obj':
                                       'CreditSupportList',
                                   'col_names':
                                       ['Independent Amount', 'Credit Rating']
                                   },
            'Posted_Threshold': {'widget': 'Table', 'description': 'Posted Threshold',
                                 'value': default['CreditSupportList'],
                                 'sub_types':
                                     [{'type': 'numeric', 'numericFormat': num_format['currency']},
                                      {'type': 'numeric', 'numericFormat': num_format['int']}],
                                 'obj':
                                     'CreditSupportList',
                                 'col_names':
                                     ['Threshold Amount', 'Credit Rating']
                                 },
            'Received_Threshold': {'widget': 'Table', 'description': 'Received Threshold',
                                   'value': default['CreditSupportList'],
                                   'sub_types':
                                       [{'type': 'numeric', 'numericFormat': num_format['currency']},
                                        {'type': 'numeric', 'numericFormat': num_format['int']}],
                                   'obj':
                                       'CreditSupportList',
                                   'col_names':
                                       ['Threshold Amount', 'Credit Rating']
                                   },
            'Minimum_Received': {'widget': 'Table', 'description': 'Minimum Received',
                                 'value': default['CreditSupportList'],
                                 'sub_types':
                                     [{'type': 'numeric', 'numericFormat': num_format['currency']},
                                      {'type': 'numeric', 'numericFormat': num_format['int']}],
                                 'obj':
                                     'CreditSupportList',
                                 'col_names':
                                     ['Minimum Amount', 'Credit Rating']
                                 },
            'Minimum_Posted': {'widget': 'Table', 'description': 'Minimum Posted',
                               'value': default['CreditSupportList'],
                               'sub_types':
                                   [{'type': 'numeric', 'numericFormat': num_format['currency']},
                                    {'type': 'numeric', 'numericFormat': num_format['int']}],
                               'obj':
                                   'CreditSupportList',
                               'col_names':
                                   ['Minimum Posted', 'Credit Rating']
                               },
            'Expiry_Date': {'widget': 'DatePicker', 'description': 'Expiry Date', 'value': default['DatePicker']},
            'Discount_Rate': {'widget': 'Text', 'description': 'Discount Rate', 'value': '', 'obj': 'Tuple'},  # tuple
            'Units': {'widget': 'Float', 'description': 'Units', 'value': 0.0},
            'Borrower_Lender': {'widget': 'Dropdown', 'description': 'Borrower Lender', 'value': 'Borrower',
                                'values': ['Borrower', 'Lender']},
            'Receive_Index_Day_Count': {'widget': 'Dropdown', 'description': 'Receive Index Day Count',
                                        'value': 'ACT_365',
                                        'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360',
                                                   'ACT_ACT_ICMA']},
            'Receive_Currency': {'widget': 'Text', 'description': 'Receive Currency', 'value': ''},  # tuple
            'Receive_Interest_Rate_Volatility': {'widget': 'Text', 'description': 'Receive Interest Rate Volatility',
                                                 'value': '', 'obj': 'Tuple'},  # tuple
            'Interest_Rate': {'widget': 'Text', 'description': 'Interest Rate', 'value': '', 'obj': 'Tuple'},  # tuple
            'Receive_Accrual_Calendars': {'widget': 'Text', 'description': 'Receive Accrual Calendars', 'value': ''},
            # tuple
            'Equity_Volatility': {'widget': 'Text', 'description': 'Equity Volatility', 'value': '', 'obj': 'Tuple'},
            # tuple
            'Receive_Penultimate_Coupon_Date': {'widget': 'DatePicker',
                                                'description': 'Receive Penultimate Coupon Date',
                                                'value': default['DatePicker']},
            'Receive_Compounding_Method': {'widget': 'Dropdown', 'description': 'Receive Compounding Method',
                                           'value': 'None',
                                           'values': ['None', 'Include_Margin', 'Flat', 'Exclude_Margin',
                                                      'Exponential']},
            'Rate_Calendars': {'widget': 'Text', 'description': 'Rate Calendars', 'value': ''},
            'Pay_Index_Calendars': {'widget': 'Text', 'description': 'Pay Index Calendars', 'value': ''},
            'Buy_Discount_Rate': {'widget': 'Text', 'description': 'Buy Discount Rate', 'value': '', 'obj': 'Tuple'},
            # tuple
            'First_Coupon_Date': {'widget': 'DatePicker', 'description': 'First Coupon Date',
                                  'value': default['DatePicker']},
            'Floating_Margin': {'widget': 'Float', 'description': 'Floating Margin', 'value': 0.0},
            'Receive_Principal': {'widget': 'Float', 'description': 'Receive Principal', 'value': 0.0},
            'Forecast_Rate_Swaption_Volatility': {'widget': 'Text', 'description': 'Forecast Rate Swaption Volatility',
                                                  'value': '', 'obj': 'Tuple'},  # tuple
            'Strike_Price': {'widget': 'Float', 'description': 'Strike Price', 'value': 0.0},
            'Forward_Price': {'widget': 'Float', 'description': 'Forward Price', 'value': 0.0},
            'Strike': {'widget': 'Float', 'description': 'Strike', 'value': 0.0},
            'Volume': {'widget': 'Float', 'description': 'Volume', 'value': 0.0},
            'Fixed_Price': {'widget': 'Float', 'description': 'Fixed Price', 'value': 0.0},
            'Realized_Average': {'widget': 'Float', 'description': 'Realized Average', 'value': 0.0},
            'FX_Realized_Average': {'widget': 'Float', 'description': 'FX Realized Average', 'value': 0.0},
            'Receive_Rate_Volatility': {'widget': 'Text', 'description': 'Receive Rate Volatility', 'value': '',
                                        'obj': 'Tuple'},  # tuple
            'MtM': {'widget': 'Text', 'description': 'MtM', 'value': ''},
            'Compounding_Method': {'widget': 'Dropdown', 'description': 'Compounding Method', 'value': 'None',
                                   'values': ['None', 'Include_Margin', 'Flat', 'Exclude_Margin', 'Exponential']},
            'Pay_Interest_Rate_Volatility': {'widget': 'Text', 'description': 'Pay Interest Rate Volatility',
                                             'value': '', 'obj': 'Tuple'},  # tuple
            'Pay_Currency': {'widget': 'Text', 'description': 'Pay Currency', 'value': ''},  # tuple
            'Bank': {'widget': 'Text', 'description': 'Bank', 'value': ''},  # tuple
            'Counterparty': {'widget': 'Text', 'description': 'Counterparty', 'value': ''},  # tuple
            'Credit_Support_Amounts': {'widget': 'Container', 'description': 'Credit Support Amounts',
                                       'value': {"Bank": "", "Counterparty": "", "Independent_Amount_Reference": "None",
                                                 "Independent_Amount": [], "Received_Threshold": [],
                                                 "Posted_Threshold": [], "Minimum_Received": [], "Minimum_Posted": []},
                                       'sub_fields': ['Bank', 'Counterparty', 'Independent_Amount_Reference',
                                                      'Independent_Amount', 'Received_Threshold', 'Posted_Threshold',
                                                      'Minimum_Received', 'Minimum_Posted']},
            'Collateral_Assets': {'widget': 'Container', 'description': 'Collateral Assets',
                                  'value': {"Cash_Collateral": [], "Bond_Collateral": [], "Equity_Collateral": [],
                                            "Commodity_Collateral": []},
                                  'sub_fields': ['Cash_Collateral', 'Bond_Collateral', 'Equity_Collateral',
                                                 'Commodity_Collateral']},
            'Float_Cashflows': {'widget': 'Container', 'description': 'Cashflows',
                                'value': {"Properties": [], "Compounding_Method": "None",
                                          "Averaging_Method": "Average_Interest", "Items": []},
                                'sub_fields': ['Properties', 'Compounding_Method', 'Averaging_Method', 'FloatItems']},
            'Equity_Cashflows': {'widget': 'Container', 'description': 'Cashflows',
                                'value': {"Items": []}, 'sub_fields': ['EquityItems']},
            'Fixed_Cashflows': {'widget': 'Container', 'description': 'Cashflows',
                                'value': {"Compounding": "No", "Items": []},
                                'sub_fields': ['Compounding', 'FixedItems']},
            'Fixed_Simple_Cashflows': {'widget': 'Container', 'description': 'Cashflows', 'value': {"Items": []},
                                       'sub_fields': ['FixedSimpleItems']},
            'Real_Yield_Cashflows': {'widget': 'Container', 'description': 'Cashflows', 'value': {"Items": []},
                                     'sub_fields': ['RealYieldItems']},
            'Energy_Cashflows': {'widget': 'Container', 'description': 'Payments', 'value': {"Items": []},
                                 'sub_fields': ['EnergyItems']},
            'Energy_Fixed_Cashflows': {'widget': 'Container', 'description': 'Payments', 'value': {"Items": []},
                                       'sub_fields': ['EnergyFixedItems']},

            'Payment_Offset': {'widget': 'Integer', 'description': 'Payment Offset', 'value': 0},
            'Index': {'widget': 'Text', 'description': 'Index', 'value': ''},  # tuple
            'Pay_Timing': {'widget': 'Dropdown', 'description': 'Pay Timing', 'value': 'End',
                           'values': ['End', 'Begin', 'Discounted']},
            'Receive_Day_Count': {'widget': 'Dropdown', 'description': 'Receive Day Count', 'value': 'ACT_365',
                                  'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360',
                                             'ACT_ACT_ICMA']},
            'Pay_Rate': {'widget': 'Float', 'description': 'Pay Rate', 'value': 0.0, 'obj': 'Basis'},  # tuple
            'Discount_Rate_Volatility': {'widget': 'Text', 'description': 'Discount Rate Volatility', 'value': '',
                                         'obj': 'Tuple'},  # tuple
            'Currency': {'widget': 'Text', 'description': 'Currency', 'value': ''},  # tuple
            'Forecast_Rate_Volatility': {'widget': 'Text', 'description': 'Forecast Rate Volatility', 'value': '',
                                         'obj': 'Tuple'},  # tuple
            'Receive_Reset_Type': {'widget': 'Dropdown', 'description': 'Receive Reset Type', 'value': 'Standard',
                                   'values': ['Standard', 'Advance', 'Arrears']},
            'Receive_Timing': {'widget': 'Dropdown', 'description': 'Receive Timing', 'value': 'End',
                               'values': ['End', 'Begin', 'Discounted']},
            'Payer_Receiver': {'widget': 'Dropdown', 'description': 'Payer Receiver', 'value': 'Payer',
                               'values': ['Payer', 'Receiver']},
            'Swap_Maturity_Date': {'widget': 'DatePicker', 'description': 'Swap Maturity Date',
                                   'value': default['DatePicker']},
            'Rate_Schedule': {'widget': 'Table', 'description': 'Rate Schedule', 'value': default['DateList'],
                              'sub_types':
                                  [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                   {'type': 'numeric', 'numericFormat': num_format['currency']}],
                              'obj':
                                  'DateList',
                              'col_names':
                                  ['Date', 'Amount']
                              },
            'Pay_Discount_Rate': {'widget': 'Text', 'description': 'Pay Discount Rate', 'value': '', 'obj': 'Tuple'},
            # tuple
            'Reset_Type': {'widget': 'Dropdown', 'description': 'Reset Type', 'value': 'Standard',
                           'values': ['Standard', 'Advance', 'Arrears']},
            'Pay_Index_Publication_Calendars': {'widget': 'Text', 'description': 'Pay Index Publication Calendars',
                                                'value': ''},
            'Index_Day_Count': {'widget': 'Dropdown', 'description': 'Index Day Count', 'value': 'ACT_365',
                                'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360',
                                           'ACT_ACT_ICMA']},
            'Buy_Sell': {'widget': 'Dropdown', 'description': 'Buy Sell', 'value': 'Buy', 'values': ['Buy', 'Sell']},
            'Net_Cashflows': {'widget': 'Dropdown', 'description': 'Net Cashflows',
                              'value': 'Yes', 'values': ['Yes', 'No']},
            'Settlement_Date': {'widget': 'DatePicker', 'description': 'Settlement Date',
                                'value': default['DatePicker']},
            'Payoff_Type': {'widget': 'Dropdown', 'description': 'Payoff Type', 'value': 'Standard',
                            'values': ['Standard', 'Quanto', 'Compo']},
            'Receive_Index_Publication_Calendars': {'widget': 'Text',
                                                    'description': 'Receive Index Publication Calendars', 'value': ''},
            'Receive_Fixed_Compounding': {'widget': 'Dropdown', 'description': 'Receive Fixed Compounding',
                                          'value': 'No', 'values': ['Yes', 'No']},
            'Floor_Rate': {'widget': 'Float', 'description': 'Floor Rate', 'value': 0.0},
            'Calendars': {'widget': 'Text', 'description': 'Calendars', 'value': ''},
            'Payoff_Currency': {'widget': 'Text', 'description': 'Payoff Currency', 'value': ''},  # tuple
            'Pay_Interest_Frequency': {'widget': 'Text', 'description': 'Pay Interest Frequency', 'value': '0M',
                                       'obj': 'Period'},
            'Interest_Frequency': {'widget': 'Text', 'description': 'Interest Frequency', 'value': '0M',
                                   'obj': 'Period'},
            'Is_Digital': {'widget': 'Dropdown', 'description': 'Is Digital', 'value': 'No', 'values': ['Yes', 'No']},
            'Pay_Rate_Constant': {'widget': 'Float', 'description': 'Pay Rate Constant', 'value': 0.0,
                                  'obj': 'Percent'},
            'Option_On_Forward': {'widget': 'Dropdown', 'description': 'Option On Forward', 'value': 'No',
                                  'values': ['Yes', 'No']},
            'Pay_Discount_Rate_Volatility': {'widget': 'Text', 'description': 'Pay Discount Rate Volatility',
                                             'value': '', 'obj': 'Tuple'},  # tuple
            'Accrue_Fee': {'widget': 'Dropdown', 'description': 'Accrue Fee', 'value': 'No', 'values': ['Yes', 'No']},
            'Pay_Rate_Type': {'widget': 'Dropdown', 'description': 'Pay Rate Type', 'value': 'Fixed',
                              'values': ['Fixed', 'Floating']},
            'Fixed_Compounding': {'widget': 'Dropdown', 'description': 'Fixed Compounding', 'value': 'No',
                                  'values': ['Yes', 'No']},
            'Is_Defaultable': {'widget': 'Dropdown', 'description': 'Is Defaultable', 'value': 'No',
                               'values': ['Yes', 'No']},
            'Pay_Known_Rates': {'widget': 'Table', 'description': 'Pay Known Rates', 'value': default['DateList'],
                                'sub_types':
                                    [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                     {'type': 'numeric', 'numericFormat': num_format['currency']}],
                                'obj':
                                    'DateList',
                                'col_names':
                                    ['Date', 'Amount']
                                },
            'Interest_Rate_Volatility': {'widget': 'Text', 'description': 'Interest Rate Volatility', 'value': '',
                                         'obj': 'Tuple'},  # tuple
            'Option_Type': {'widget': 'Dropdown', 'description': 'Option Type', 'value': 'Call',
                            'values': ['Call', 'Put']},
            'Receive_Rate_Type': {'widget': 'Dropdown', 'description': 'Receive Rate Type', 'value': 'Floating',
                                  'values': ['Fixed', 'Floating']},
            'FX_Volatility': {'widget': 'Text', 'description': 'FX Volatility', 'value': '', 'obj': 'Tuple'},  # tuple
            'Receive_Calendars': {'widget': 'Text', 'description': 'Receive Calendars', 'value': ''},
            'Payment_Interval': {'widget': 'Text', 'description': 'Payment Interval', 'value': '3M', 'obj': 'Period'},
            'Recovery_Rate': {'widget': 'Text', 'description': 'Recovery Rate', 'value': '', 'obj': 'Tuple'},  # tuple
            'Settlement_Amount': {'widget': 'Float', 'description': 'Settlement Amount', 'value': 0.0},
            'Receive_Rate': {'widget': 'Text', 'description': 'Receive Rate', 'value': '', 'obj': 'Tuple'},  # tuple
            'Index_Offset': {'widget': 'Integer', 'description': 'Index Offset', 'value': 0},
            'Tenor': {'widget': 'Text', 'description': 'Tenor', 'value': '3M', 'obj': 'Period'},
            'Settlement_Set': {'widget': 'Dropdown', 'description': 'Settlement Set', 'value': 'No',
                               'values': ['Yes', 'No']},
            'Issuer': {'widget': 'Text', 'description': 'Issuer', 'value': '', 'obj': 'Tuple'},  # tuple
            'Receive_Known_Rates': {'widget': 'Table', 'description': 'Receive Known Rates',
                                    'value': default['DateList'],
                                    'sub_types':
                                        [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                         {'type': 'numeric', 'numericFormat': num_format['currency']}],
                                    'obj':
                                        'DateList',
                                    'col_names':
                                        ['Date', 'Amount']
                                    },
            'Equity': {'widget': 'Text', 'description': 'Equity', 'value': '', 'obj': 'Tuple'},  # tuple
            'Receive_Discount_Rate': {'widget': 'Text', 'description': 'Receive Discount Rate', 'value': '',
                                      'obj': 'Tuple'},  # tuple
            'Pay_Penultimate_Coupon_Date': {'widget': 'DatePicker', 'description': 'Pay Penultimate Coupon Date',
                                            'value': default['DatePicker']},
            'Pay_Day_Count': {'widget': 'Dropdown', 'description': 'Pay Day Count', 'value': 'ACT_365',
                              'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']},
            'Settlement_Amount_Is_Clean': {'widget': 'Dropdown', 'description': 'Settlement Amount Is Clean',
                                           'value': 'Yes', 'values': ['Yes', 'No']},
            'Sell_Discount_Rate': {'widget': 'Text', 'description': 'Sell Discount Rate', 'value': '', 'obj': 'Tuple'},
            # tuple
            'Discount_Rate_Cap_Volatility': {'widget': 'Text', 'description': 'Discount Rate Cap Volatility',
                                             'value': '', 'obj': 'Tuple'},  # tuple
            'Forecast_Rate': {'widget': 'Text', 'description': 'Forecast Rate', 'value': '', 'obj': 'Tuple'},  # tuple
            'Index_Tenor': {'widget': 'Text', 'description': 'Index Tenor', 'value': '0M', 'obj': 'Period'},
            'Receive_Margin': {'widget': 'Float', 'description': 'Receive Margin', 'value': 0.0},
            'Index_Calendars': {'widget': 'Text', 'description': 'Index Calendars', 'value': ''},
            'Known_Rates': {'widget': 'Table', 'description': 'Known Rates', 'value': default['DateList'],
                            'sub_types':
                                [{'type': 'date', 'dateFormat': 'YYYY-MM-DD'},
                                 {'type': 'numeric', 'numericFormat': num_format['currency']}],
                            'obj':
                                'DateList',
                            'col_names':
                                ['Date', 'Amount']
                            },
            'Description': {'widget': 'Text', 'description': 'Description', 'value': ''},
            'Tags': {'widget': 'Text', 'description': 'Tags', 'value': ''},
            'Receive_Index_Offset': {'widget': 'Integer', 'description': 'Receive Index Offset', 'value': 0},
            'Receive_Payment_Calendars': {'widget': 'Text', 'description': 'Receive Payment Calendars', 'value': ''},
            'Forecast_Rate_Cap_Volatility': {'widget': 'Text', 'description': 'Forecast Rate Cap Volatility',
                                             'value': '', 'obj': 'Tuple'},  # tuple
            'Receive_Interest_Frequency': {'widget': 'Text', 'description': 'Receive Interest Frequency', 'value': '0M',
                                           'obj': 'Period'},
            'In_Arrears': {'widget': 'Dropdown', 'description': 'In Arrears', 'value': 'No', 'values': ['Yes', 'No']},
            'Pay_Margin': {'widget': 'Float', 'description': 'Pay Margin', 'value': 0.0},
            'Accrual_Calendars': {'widget': 'Text', 'description': 'Accrual Calendars', 'value': ''},
            'Repo_Rate': {'widget': 'Text', 'description': 'Repo Rate', 'value': '', 'obj': 'Tuple'},  # tuple
            'Name': {'widget': 'Text', 'description': 'Name', 'value': ''},
            'Principal_Exchange': {'widget': 'Dropdown', 'description': 'Principal Exchange', 'value': 'None',
                                   'values': ['None', 'Start', 'Maturity', 'Start_Maturity']},
            'Rate_Constant': {'widget': 'Float', 'description': 'Rate Constant', 'value': 0.0, 'obj': 'Percent'},
            'Accrual_Day_Count': {'widget': 'Dropdown', 'description': 'Accrual Day Count', 'value': 'ACT_365',
                                  'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360',
                                             'ACT_ACT_ICMA']},
            'Receive_Discount_Rate_Volatility': {'widget': 'Text', 'description': 'Receive Discount Rate Volatility',
                                                 'value': '', 'obj': 'Tuple'},  # tuple
            'Index_Publication_Calendars': {'widget': 'Text', 'description': 'Index Publication Calendars',
                                            'value': ''},
            'Forward_Price_Date': {'widget': 'DatePicker', 'description': 'Forward Price Date',
                                   'value': default['DatePicker']},
            'Pay_Fixed_Rate': {'widget': 'Float', 'description': 'Pay Fixed Rate', 'value': 0.0},
            'Receive_Index_Frequency': {'widget': 'Text', 'description': 'Receive Index Frequency', 'value': '0M',
                                        'obj': 'Period'},
            'Receive_Index_Calendars': {'widget': 'Text', 'description': 'Receive Index Calendars', 'value': ''},
            'Maturity_Date': {'widget': 'DatePicker', 'description': 'Maturity Date',
                              'value': default['DatePicker']},
            'Receive_Interest_Rate': {'widget': 'Text', 'description': 'Receive Interest Rate', 'value': '',
                                      'obj': 'Tuple'},  # tuple
            'Pay_Index_Offset': {'widget': 'Integer', 'description': 'Pay Index Offset', 'value': 0},
            'Day_Count': {'widget': 'Dropdown', 'description': 'Day Count', 'value': 'ACT_365',
                          'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']},
            'Receive_First_Coupon_Date': {'widget': 'DatePicker', 'description': 'Receive First Coupon Date',
                                          'value': default['DatePicker']},
            'Pay_Accrual_Calendars': {'widget': 'Text', 'description': 'Pay Accrual Calendars', 'value': ''},
            'Receive_Rate_Constant': {'widget': 'Float', 'description': 'Receive Rate Constant', 'value': 0.0,
                                      'obj': 'Percent'},
            'Receive_Fixed_Rate': {'widget': 'Float', 'description': 'Receive Fixed Rate', 'value': 0.0},
            'Pay_Payment_Calendars': {'widget': 'Text', 'description': 'Pay Payment Calendars', 'value': ''},
            'Swap_Rate': {'widget': 'Float', 'description': 'Swap Rate', 'value': 0.0},
            'FRA_Rate': {'widget': 'Float', 'description': 'FRA Rate', 'value': 0.0},
            'Principal': {'widget': 'Float', 'description': 'Principal', 'value': 0.0}
        }
    }
}
