if __name__=='__main__':
    import argparse
    import os
    import re
    import torch
    import riskflow as rf

    parser = argparse.ArgumentParser(description='Run the solve_hedge test job.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random_Seed — seeds the outer Sobol and the torch global '
                             'RNG behind the inner-MC Student-t draws. Vary it across '
                             'instances for independent V_0 estimates.')
    parser.add_argument('--multi-seed-count', type=int, default=1,
                        help='Solver.Multi_Seed_Count — inner-MC re-seed repeats per run.')
    args = parser.parse_args()

    cx = rf.Context()

    json = '''
    {
        "Calc": {
            "Calculation": {
                "Object": "HedgeMonteCarlo",
                "Time_Grid": "0d 1d(1d)",
                "Base_Date": {
                    ".Timestamp": "2026-04-10"
                },
                "Simulation_Batches": 1,
                "Batch_Size": 2048,
                "Random_Seed": 42,
                "Currency": "USD",
                "Calendar": "Chicago",
                "Execution_Mode": "solve_hedge",
                "Inner_MC_Enabled": "Yes",
                "Inner_Sub_Batch": 512,
                "Hedging_Problem": {
                    "History_Lookback_Business_Days": 30,
                    "Tradable_Instruments": {
                        "CommodityFutureDeal": {
                            "PL_APR_2026": {
                                 "Maturity_Date": {".Timestamp": "2026-04-29"},
                                 "Currency": "USD",
                                 "Carry": "PLATINUM_CARRY",
                                 "Repo_Rate": "USD-SOFR",
                                 "Implied_Basis": "LME_CME",
                                 "Contract_Size": 50
                            },
                            "PL_JUL_2026": {
                                 "Maturity_Date": {".Timestamp": "2026-07-29"},
                                 "Currency": "USD",
                                 "Carry": "PLATINUM_CARRY",
                                 "Repo_Rate": "USD-SOFR",
                                 "Implied_Basis": "LME_CME",
                                 "Contract_Size": 50
                            },
                            "PL_OCT_2026": {
                                 "Maturity_Date": {".Timestamp": "2026-10-29"},
                                 "Currency": "USD",
                                 "Carry": "PLATINUM_CARRY",
                                 "Repo_Rate": "USD-SOFR",
                                 "Implied_Basis": "LME_CME",
                                 "Contract_Size": 50
                            }
                        },
                        "CashAccountDeal": {
                            "USD_CASH": {
                                "Currency": "USD",
                                "Investment_Horizon": {".Timestamp": "2026-08-05"},
                                "Discount_Rate": "USD-SOFR"
                            }
                        }
                    },
                    "Portfolio_State": {
                        "Positions": {
                            "PL_APR_2026": 0,
                            "PL_JUL_2026": 0,
                            "PL_OCT_2026": 0
                        },
                        "Cash_Balances": {
                            "USD_CASH": 125000.0
                        },
                         "Settlement_Prices": {
                            "PL_APR_2026": 2026.1718,
                            "PL_JUL_2026": 2047.0685,
                            "PL_OCT_2026": 2070.7498
                        },
                        "Margin_Balances": {
                            "USD_CASH": 0.0
                        },
                        "Initial_Margin": {
                            "PL_APR_2026": {
                                "Method": "per_contract",
                                "Amount": 8500.0
                            },
                            "PL_JUL_2026": {
                                "Method": "per_contract",
                                "Amount": 9000.0
                            },
                            "PL_OCT_2026": {
                                "Method": "per_contract",
                                "Amount": 9500.0
                            }
                        },
                        "Spot_Price_History": {
                            "CommodityPrice.PLATINUM_LME": {
                                "Dates": [
                                    {".Timestamp": "2026-02-26"},
                                    {".Timestamp": "2026-02-27"},
                                    {".Timestamp": "2026-03-02"},
                                    {".Timestamp": "2026-03-03"},
                                    {".Timestamp": "2026-03-04"},
                                    {".Timestamp": "2026-03-05"},
                                    {".Timestamp": "2026-03-06"},
                                    {".Timestamp": "2026-03-09"},
                                    {".Timestamp": "2026-03-10"},
                                    {".Timestamp": "2026-03-11"},
                                    {".Timestamp": "2026-03-12"},
                                    {".Timestamp": "2026-03-13"},
                                    {".Timestamp": "2026-03-16"},
                                    {".Timestamp": "2026-03-17"},
                                    {".Timestamp": "2026-03-18"},
                                    {".Timestamp": "2026-03-19"},
                                    {".Timestamp": "2026-03-20"},
                                    {".Timestamp": "2026-03-23"},
                                    {".Timestamp": "2026-03-24"},
                                    {".Timestamp": "2026-03-25"},
                                    {".Timestamp": "2026-03-26"},
                                    {".Timestamp": "2026-03-27"},
                                    {".Timestamp": "2026-03-30"},
                                    {".Timestamp": "2026-03-31"},
                                    {".Timestamp": "2026-04-01"},
                                    {".Timestamp": "2026-04-02"},
                                    {".Timestamp": "2026-04-06"},
                                    {".Timestamp": "2026-04-07"},
                                    {".Timestamp": "2026-04-08"},
                                    {".Timestamp": "2026-04-09"}
                                ],
                                "Prices": [
                                    2365.5,2272.5,
                                    2351,2123.5,2167.5,2164.5,2119,
                                    2121.5,2231,2186.5,2172.5,2076,
                                    2083,2149.5,2087,1939.5,1977,
                                    1840.5,1896,1951.5,1877.5,1858.5,
                                    1926.5, 1915.5, 2003.5, 1964.5, 
                                    1916.5, 1962.5, 2053.0, 2028.5
                                ]
                            }
                        }
                    },
                    "Objective": {
                        "Object": "AsymmetricUtility_Symlog",
                        "Floor_Penalty": 50.0,
                        "Surplus_Reward": 1.0,
                        "Power": 1.0,
                        "Expiry_Penalty": 1.0,
                        "Expiry_Threshold_Days": 4.0,
                        "Post_Deal_Trade_Penalty": 1.0,
                        "Position_Bounds_Penalty": 1.0
                    },
                    "Liabilities": {
                        "FloatingEnergyDeal": {
                            "PLAT_JUL29": {
                                "Currency": "USD",
                                "Sampling_Type": "USD",
                                "FX_Sampling_Type": "USD",
                                "Discount_Rate": "USD-SOFR",
                                "Commodity": "PLATINUM_LME",
                                "Reference_Type": "PLATINUM",
                                "Payer_Receiver": "Receiver",
                                "Payments": {
                                    "Items": [
                                        {
                                            "Payment_Date": {
                                                ".Timestamp": "2026-08-05"
                                            },
                                            "Period_Start": {
                                                ".Timestamp": "2026-07-01"
                                            },
                                            "Period_End": {
                                                ".Timestamp": "2026-07-31"
                                            },                                            
                                            "Volume": 2500.0,
                                            "Fixed_Basis": -2055.0,
                                            "Price_Multiplier": 1.0,
                                            "FX_Period_Start": {
                                                ".Timestamp": "2026-07-01"
                                            },
                                            "Realized_Average": 0.0,
                                            "FX_Period_End": {
                                                ".Timestamp": "2026-07-31"
                                            },
                                            "FX_Realized_Average": 0.0
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    "Evaluator": {
                        "Object": "DynamicHedgeEvaluator",
                        "Accounting_Mode": "futures",
                        "Cash_Instruments": [
                            "USD_CASH"
                        ],
                        "Force_Flat_At_End": true,
                        "Transaction_Cost_Per_Unit": 0.0,
                        "Bid_Offer_Spread_Bps": 10.0,
                        "Position_Limits": {
                            "PL_APR_2026": {"Min_Position": -50, "Max_Position": 0},
                            "PL_JUL_2026": {"Min_Position": -50, "Max_Position": 0},
                            "PL_OCT_2026": {"Min_Position": -50, "Max_Position": 0}
                        },
                        "Total_Position_Abs_Limit": 50
                    },
                    "Solver": {
                        "Object": "LsmDpSolver",
                        "Training_Action_Grid_Levels_Per_Axis": 11,
                        "Training_Action_Chunk_Size": 64,
                        "Decision_Action_Search": "fine_grid",
                        "Decision_Action_Grid_Levels_Per_Axis": 51,
                        "Turnover_Cost_Padding_Multiplier": 1.5,
                        "Range_Projection_Alpha": 0.001,
                        "Value_Fn": {
                            "MLP_Hidden": [64, 64, 64],
                            "MLP_Activation": "gelu",
                            "MLP_Train_Steps_Per_Solve": 0,
                            "MLP_Adam_LR": 1.0e-3,
                            "MLP_Final_Init_Scale": 0.0
                        },
                        "Include_Dynamic_Features_In_Value_Inputs": false,
                        "Multi_Seed_Count": 1,
                        "Run_Hindsight_Diagnostic": true,
                        "Run_Mpc_Comparison": true,
                        "Run_Textbook_Benchmark": true
                    }
                }
            },
            "MergeMarketData": {
                "MarketDataFile": "./data/MarketDataRF_platinum_calibrated.json",
                "ExplicitMarketData": {
                    "System Parameters": {
                        "Base_Currency": "USD"
                    },
                    "Price Factors": {
                        "FxRate.USD": {
                            "Domestic_Currency": "",
                            "Interest_Rate": "USD-SOFR",
                            "Spot": 1.0
                        },
                        "FxRate.ZAR": {
                            "Domestic_Currency": "USD",
                            "Interest_Rate": "ZAR-SWAP.ZAR-USD-BASIS",
                            "Spot": 0.06185898625493325
                        },
                        "CommodityPriceVol.PLATINUM": {
                            "Surface": {
                                ".Curve": {
                                    "meta": [
                                        2,
                                        "Default"
                                    ],
                                    "data": [
                                        [
                                            1,
                                            0.0821917808219178,
                                            0.385
                                        ],
                                        [
                                            1.1,
                                            0.16712328767123289,
                                            0.37
                                        ],
                                        [
                                            0.9,
                                            0.2493150684931507,
                                            0.36
                                        ],
                                        [
                                            0.8,
                                            0.5013698630136987,
                                            0.33
                                        ],
                                        [
                                            0.7,
                                            1.0,
                                            0.315
                                        ]
                                    ]
                                }
                            },
                            "Currency": "USD"
                        },
                        "CommodityPrice.PLATINUM_LME": {
                            "Currency": "USD",
                            "Interest_Rate": "USD-SOFR",
                            "Spot": 2055.0,
                            "Property_Aliases": ""
                        },
                        "CommodityBasis.LME_CME": {
                            "Spot": -32.5375,
                            "Observed_Commodity": "PLATINUM_LME"
                        },
                        "ReferencePrice.PLATINUM": {
                            "Fixing_Curve": {
                                ".Curve": {
                                    "meta": [],
                                    "data": [
                                        [
                                            46135,
                                            46135
                                        ],
                                        [
                                            47148,
                                            47148
                                        ]
                                    ]
                                }
                            },
                            "ForwardPrice": "PLATINUM",
                            "Property_Aliases": ""
                        },
                        "ForwardPrice.PLATINUM": {
                            "Curve": {
                                ".Curve": {
                                    "meta": [],
                                    "data": [
                                        [
                                            46125,
                                            2063.229999639941
                                        ],
                                        [
                                            46141,
                                            2036.000000176416
                                        ],
                                        [
                                            46233,
                                            2070.300000120793
                                        ],
                                        [
                                            46277,
                                            2078.8098901120476
                                        ]
                                    ]
                                }
                            },
                            "Currency": "USD",
                            "Fixings": "",
                            "Property_Aliases": ""
                        },
                        "ForwardRate.PLATINUM_CARRY": {
                            "Currency": "USD",
                            "Curve": {
                                ".Curve": {
                                    "meta": [],
                                    "data": [
                                        [46141, -0.0016243231840737684],
                                        [46232, 0.0031160703062786694],
                                        [46324, 0.005675358368189906]
                                    ]
                                }
                            }
                        },
                        "InterestRate.ZAR-SWAP": {
                            "Day_Count": "ACT_365",
                            "Currency": "ZAR",
                            "Curve": {
                                ".Curve": {
                                    "meta": [],
                                    "data": [
                                        [
                                            0.00821917808219178,
                                            0.06662175643011484
                                        ],
                                        [
                                            0.08493150684931507,
                                            0.06673054371954815
                                        ],
                                        [
                                            0.2493150684931507,
                                            0.06710550489097832
                                        ],
                                        [
                                            0.336986301369863,
                                            0.0672123261686964
                                        ],
                                        [
                                            0.4191780821917808,
                                            0.06791312946053238
                                        ],
                                        [
                                            0.5068493150684932,
                                            0.06795993863792772
                                        ],
                                        [
                                            0.5863013698630137,
                                            0.06822402151580623
                                        ],
                                        [
                                            0.6684931506849315,
                                            0.06855787487002473
                                        ],
                                        [
                                            0.7589041095890411,
                                            0.06853269137460189
                                        ],
                                        [
                                            0.8383561643835616,
                                            0.06866309629781273
                                        ],
                                        [
                                            0.915068493150685,
                                            0.06882526928495039
                                        ],
                                        [
                                            1.0054794520547946,
                                            0.06886713202186877
                                        ],
                                        [
                                            1.2547945205479452,
                                            0.0689136644198827
                                        ],
                                        [
                                            1.5068493150684932,
                                            0.06886177019382163
                                        ],
                                        [
                                            1.7561643835616439,
                                            0.06872687519591134
                                        ],
                                        [
                                            2.0054794520547949,
                                            0.06853382315832583
                                        ],
                                        [
                                            3.0027397260273975,
                                            0.06850467582010055
                                        ],
                                        [
                                            4.002739726027397,
                                            0.06916589986791735
                                        ],
                                        [
                                            5.002739726027397,
                                            0.07012306945464186
                                        ],
                                        [
                                            6.010958904109589,
                                            0.07145818200365649
                                        ],
                                        [
                                            7.010958904109589,
                                            0.07289356176760638
                                        ],
                                        [
                                            8.005479452054795,
                                            0.07431323704452053
                                        ],
                                        [
                                            9.005479452054795,
                                            0.0759240340036614
                                        ],
                                        [
                                            10.008219178082192,
                                            0.07716542980986775
                                        ],
                                        [
                                            12.013698630136986,
                                            0.0795143458718339
                                        ],
                                        [
                                            15.01095890410959,
                                            0.08129308637724983
                                        ],
                                        [
                                            20.013698630136987,
                                            0.08134391952133739
                                        ],
                                        [
                                            25.016438356164384,
                                            0.07976091282904245
                                        ],
                                        [
                                            30.02191780821918,
                                            0.07820908529482024
                                        ]
                                    ]
                                }
                            },
                            "Sub_Type": null
                        },
                        "InterestRate.ZAR-SWAP.ZAR-USD-BASIS": {
                            "Day_Count": "ACT_365",
                            "Currency": "ZAR",
                            "Curve": {
                                ".Curve": {
                                    "meta": [],
                                    "data": [
                                        [
                                            0.00821917808219178,
                                            -0.006088812061550271
                                        ],
                                        [
                                            0.010958904109589042,
                                            -0.0057403710796361949
                                        ],
                                        [
                                            0.0136986301369863,
                                            -0.005546463754933999
                                        ],
                                        [
                                            0.030136986301369865,
                                            -0.0045982242954521819
                                        ],
                                        [
                                            0.049315068493150687,
                                            -0.0042041238841820098
                                        ],
                                        [
                                            0.08493150684931507,
                                            -0.0038222640257019188
                                        ],
                                        [
                                            0.09315068493150686,
                                            -0.0037675280845574784
                                        ],
                                        [
                                            0.18082191780821919,
                                            -0.0037618534557495588
                                        ],
                                        [
                                            0.2493150684931507,
                                            -0.003556727015579783
                                        ],
                                        [
                                            0.2602739726027397,
                                            -0.0035420873107818164
                                        ],
                                        [
                                            0.336986301369863,
                                            -0.0036735011997751506
                                        ],
                                        [
                                            0.3452054794520548,
                                            -0.0036804431946149548
                                        ],
                                        [
                                            0.4191780821917808,
                                            -0.0036298925502774028
                                        ],
                                        [
                                            0.4301369863013699,
                                            -0.0036162231601520879
                                        ],
                                        [
                                            0.5068493150684932,
                                            -0.0034815467255827627
                                        ],
                                        [
                                            0.5123287671232877,
                                            -0.0034758782489398567
                                        ],
                                        [
                                            0.5863013698630137,
                                            -0.0034652368269893626
                                        ],
                                        [
                                            0.6027397260273972,
                                            -0.0034752125772520155
                                        ],
                                        [
                                            0.6684931506849315,
                                            -0.003573760579151153
                                        ],
                                        [
                                            0.6794520547945205,
                                            -0.0035770110489233427
                                        ],
                                        [
                                            0.7589041095890411,
                                            -0.0034268258663783664
                                        ],
                                        [
                                            0.7643835616438356,
                                            -0.0034195372287975647
                                        ],
                                        [
                                            0.8383561643835616,
                                            -0.003385194997047064
                                        ],
                                        [
                                            0.852054794520548,
                                            -0.003384560076064011
                                        ],
                                        [
                                            0.915068493150685,
                                            -0.003407296337288951
                                        ],
                                        [
                                            0.9315068493150684,
                                            -0.003407252577504183
                                        ],
                                        [
                                            1.0054794520547946,
                                            -0.003347722335113601
                                        ],
                                        [
                                            1.010958904109589,
                                            -0.0033433001764465316
                                        ],
                                        [
                                            1.2547945205479452,
                                            -0.0032002275881457405
                                        ],
                                        [
                                            1.5068493150684932,
                                            -0.003123755423156807
                                        ],
                                        [
                                            1.7561643835616439,
                                            -0.003088185632091525
                                        ],
                                        [
                                            2.0054794520547949,
                                            -0.0030610462386982536
                                        ],
                                        [
                                            2.0136986301369865,
                                            -0.0030599259039252037
                                        ],
                                        [
                                            3.0027397260273975,
                                            -0.0030649035592468234
                                        ],
                                        [
                                            3.0191780821917808,
                                            -0.0030650040370471117
                                        ],
                                        [
                                            4.002739726027397,
                                            -0.0030681888888920568
                                        ],
                                        [
                                            4.019178082191781,
                                            -0.003068208411098214
                                        ],
                                        [
                                            5.002739726027397,
                                            -0.003067145781247435
                                        ],
                                        [
                                            5.013698630136986,
                                            -0.003067481389566029
                                        ],
                                        [
                                            6.010958904109589,
                                            -0.0031315107223634649
                                        ],
                                        [
                                            6.016438356164383,
                                            -0.003131508711683731
                                        ],
                                        [
                                            7.010958904109589,
                                            -0.0030653983509497525
                                        ],
                                        [
                                            7.016438356164383,
                                            -0.003065563729011303
                                        ],
                                        [
                                            8.005479452054795,
                                            -0.0031933544639186769
                                        ],
                                        [
                                            8.016438356164385,
                                            -0.00319441973763189
                                        ],
                                        [
                                            9.005479452054795,
                                            -0.0032551689947788316
                                        ],
                                        [
                                            9.021917808219179,
                                            -0.0032567690997532124
                                        ],
                                        [
                                            10.008219178082192,
                                            -0.003390648793367798
                                        ],
                                        [
                                            10.01917808219178,
                                            -0.0033919850453048435
                                        ],
                                        [
                                            12.013698630136986,
                                            -0.00357668383876783
                                        ],
                                        [
                                            15.01095890410959,
                                            -0.0036459545680471718
                                        ],
                                        [
                                            15.027397260273972,
                                            -0.003645644899747766
                                        ],
                                        [
                                            20.013698630136987,
                                            -0.003627626997062497
                                        ],
                                        [
                                            25.016438356164384,
                                            -0.003633262416083985
                                        ],
                                        [
                                            30.02191780821918,
                                            -0.0036389144466224267
                                        ]
                                    ]
                                }
                            },
                            "Sub_Type": "BasisSpread"
                        },
                        "ForwardPriceSample.USD": {
                            "Offset": 0,
                            "Holiday_Calendar": "New York",
                            "Sampling_Convention": "ForwardPriceSampleDaily"
                        },
                        "InterestRate.USD-SOFR": {
                            "Day_Count": "ACT_365",
                            "Currency": "USD",
                            "Curve": {
                                ".Curve": {
                                    "meta": [],
                                    "data": [
                                        [
                                            0.0027397260273972605,
                                            0.03670093258093306
                                        ],
                                        [
                                            0.03561643835616438,
                                            0.03684780068875626
                                        ],
                                        [
                                            0.0547945205479452,
                                            0.03682433403267367
                                        ],
                                        [
                                            0.07397260273972603,
                                            0.036994182911425647
                                        ],
                                        [
                                            0.0958904109589041,
                                            0.036911717809208637
                                        ],
                                        [
                                            0.1780821917808219,
                                            0.03699905853506198
                                        ],
                                        [
                                            0.26575342465753429,
                                            0.037017693558552989
                                        ],
                                        [
                                            0.3452054794520548,
                                            0.037003190390053296
                                        ],
                                        [
                                            0.43561643835616439,
                                            0.036991073248304629
                                        ],
                                        [
                                            0.5205479452054794,
                                            0.036973071190586338
                                        ],
                                        [
                                            0.6,
                                            0.03694666668810367
                                        ],
                                        [
                                            0.684931506849315,
                                            0.0369220252768245
                                        ],
                                        [
                                            0.7671232876712328,
                                            0.0368685687506485
                                        ],
                                        [
                                            0.8493150684931506,
                                            0.03681258437677016
                                        ],
                                        [
                                            0.9260273972602739,
                                            0.03675465667599013
                                        ],
                                        [
                                            1.0164383561643836,
                                            0.03669192082050027
                                        ],
                                        [
                                            1.5178082191780822,
                                            0.03626190256307585
                                        ],
                                        [
                                            2.0136986301369865,
                                            0.03571651310001294
                                        ],
                                        [
                                            3.0136986301369865,
                                            0.0352824410861282
                                        ],
                                        [
                                            4.013698630136986,
                                            0.03536873686535984
                                        ],
                                        [
                                            5.019178082191781,
                                            0.035735225391713146
                                        ],
                                        [
                                            6.021917808219178,
                                            0.036243817285599878
                                        ],
                                        [
                                            7.019178082191781,
                                            0.03678733979015485
                                        ],
                                        [
                                            8.016438356164385,
                                            0.03731658805075925
                                        ],
                                        [
                                            9.016438356164385,
                                            0.03784577136602518
                                        ],
                                        [
                                            10.024657534246576,
                                            0.03837317751704552
                                        ],
                                        [
                                            12.024657534246576,
                                            0.039411821828701228
                                        ],
                                        [
                                            15.021917808219179,
                                            0.040761993401692977
                                        ],
                                        [
                                            20.024657534246577,
                                            0.04185924657730623
                                        ],
                                        [
                                            25.027397260273973,
                                            0.04177606662203917
                                        ],
                                        [
                                            30.03287671232877,
                                            0.041071980400768207
                                        ]
                                    ]
                                }
                            },
                            "Sub_Type": null
                        }
                    }
                }
            },
            "CalendDataFile": "./data/AACalendars.cal"
        }
    }
    '''

    json, n = re.subn(r'"Random_Seed":\s*\d+', f'"Random_Seed": {args.seed}', json)
    assert n == 1, f'expected exactly one Random_Seed in the config, found {n}'
    json, n = re.subn(r'"Multi_Seed_Count":\s*\d+',
                      f'"Multi_Seed_Count": {args.multi_seed_count}', json)
    assert n == 1, f'expected exactly one Multi_Seed_Count in the config, found {n}'

    cx.load_json((json, 'hedge_test.json'))

    calc, result = cx.run_job()
    if torch.cuda.is_available():
        print(f'[peak GPU allocated] {torch.cuda.max_memory_allocated() // (1024 * 1024)} MB '
              f'(reserved {torch.cuda.max_memory_reserved() // (1024 * 1024)} MB)', flush=True)

    print(result.evaluation_summary)