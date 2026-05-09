import os


if __name__=='__main__':
    import riskflow as rf
    from riskflow.structured_policy import save_policy_artifact

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
                "Simulation_Batches": 4,
                "Batch_Size": 2048,
                "Random_Seed": 42,
                "Currency": "USD",
                "Calendar": "Chicago",
                "Execution_Mode": "optimize_policy",
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
                        "Object": "TerminalFloorThenSurplusUtility",
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
                    "Policy": {
                        "Object": "StructuredRebalancePolicy",
                        "Action_Space": {
                            "Instrument_Order": [
                                "PL_APR_2026", "PL_JUL_2026", "PL_OCT_2026"
                            ],
                            "Trade_Deltas": [
                                [-50, -45, -40, -35, -30, -25, -20, -15,
                                 -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 15, 20, 25, 30, 35, 40, 45, 50],
                                [-50, -45, -40, -35, -30, -25, -20, -15,
                                 -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 15, 20, 25, 30, 35, 40, 45, 50],
                                [-50, -45, -40, -35, -30, -25, -20, -15,
                                 -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 15, 20, 25, 30, 35, 40, 45, 50]
                            ]
                        },
                        "Model": {
                            "Object": "EntityTransformer",
                            "Token_Dim": 64,
                            "Emb_Dim": 8,
                            "N_Heads": 4,
                            "N_Layers": 2
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
                    "Optimizer": {
                        "Object": "PPO",
                        "Epochs": 80,
                        "PPO_Epochs": 4,
                        "Minibatch_Size": 8192,
                        "Gamma": 1.0,
                        "GAE_Lambda": 0.995,
                        "Learning_Rate": 0.0003,
                        "LR_Schedule": "cosine",
                        "LR_Min": 1.0e-5,
                        "Clip_Eps": 0.2,
                        "Value_Coef": 0.1,
                        "Entropy_Coef": 0.001,
                        "Max_Grad_Norm": 0.5,
                        "Reward_Scale": 1.0e-6,
                        "Dense_Tracking_Reward_Scale": 2.0,
                        "Dense_Reward_Mode": "asymmetric",
                        "Anchor_Beta": 0.0,
                        "Anchor_Beta_Floor": 0.0,
                        "Anchor_Anneal_Epochs": 0,
                        "Anchor_Bin_Sharpness": 2.0,
                        "Anchor_Target": "delta1_jul",
                        "CVaR_Alpha": 0.0,
                        "CVaR_Lambda": 0.0,
                        "Value_Loss_Asym_Weight": 1.0,
                        "Entropy_Floor_H_Min": 0.0,
                        "Entropy_Floor_Coef": 0.0,
                        "Validation_Fraction": 0.25,
                        "Validation_Min_Batch": 512,
                        "Decision_Interval_Curriculum": [
                            {"Start_Epoch": 1, "End_Epoch": 80, "Interval_Business_Days": 1}
                        ],
                        "Seed": 42
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
                                            46126,
                                            2061.5281246734709
                                        ],
                                        [
                                            46127,
                                            2059.8262497070006
                                        ],
                                        [
                                            46128,
                                            2058.12437474053
                                        ],
                                        [
                                            46129,
                                            2056.42249977406
                                        ],
                                        [
                                            46130,
                                            2054.7206248075897
                                        ],
                                        [
                                            46131,
                                            2053.018749841119
                                        ],
                                        [
                                            46132,
                                            2051.316874874649
                                        ],
                                        [
                                            46133,
                                            2049.6149999081787
                                        ],
                                        [
                                            46134,
                                            2047.913124941708
                                        ],
                                        [
                                            46135,
                                            2046.211249975238
                                        ],
                                        [
                                            46136,
                                            2044.5093750087677
                                        ],
                                        [
                                            46137,
                                            2042.807500042297
                                        ],
                                        [
                                            46138,
                                            2041.1056250758268
                                        ],
                                        [
                                            46139,
                                            2039.4037501093567
                                        ],
                                        [
                                            46140,
                                            2037.7018751428864
                                        ],
                                        [
                                            46141,
                                            2036.000000176416
                                        ],
                                        [
                                            46142,
                                            2036.372826262768
                                        ],
                                        [
                                            46143,
                                            2036.74565234912
                                        ],
                                        [
                                            46144,
                                            2037.1184784354719
                                        ],
                                        [
                                            46145,
                                            2037.4913045218237
                                        ],
                                        [
                                            46146,
                                            2037.8641306081757
                                        ],
                                        [
                                            46147,
                                            2038.2369566945277
                                        ],
                                        [
                                            46148,
                                            2038.6097827808797
                                        ],
                                        [
                                            46149,
                                            2038.9826088672313
                                        ],
                                        [
                                            46150,
                                            2039.3554349535835
                                        ],
                                        [
                                            46151,
                                            2039.7282610399353
                                        ],
                                        [
                                            46152,
                                            2040.1010871262873
                                        ],
                                        [
                                            46153,
                                            2040.4739132126388
                                        ],
                                        [
                                            46154,
                                            2040.8467392989913
                                        ],
                                        [
                                            46155,
                                            2041.219565385343
                                        ],
                                        [
                                            46156,
                                            2041.5923914716948
                                        ],
                                        [
                                            46157,
                                            2041.9652175580469
                                        ],
                                        [
                                            46158,
                                            2042.3380436443989
                                        ],
                                        [
                                            46159,
                                            2042.7108697307506
                                        ],
                                        [
                                            46160,
                                            2043.0836958171024
                                        ],
                                        [
                                            46161,
                                            2043.4565219034546
                                        ],
                                        [
                                            46162,
                                            2043.8293479898064
                                        ],
                                        [
                                            46163,
                                            2044.2021740761582
                                        ],
                                        [
                                            46164,
                                            2044.5750001625102
                                        ],
                                        [
                                            46165,
                                            2044.9478262488622
                                        ],
                                        [
                                            46166,
                                            2045.320652335214
                                        ],
                                        [
                                            46167,
                                            2045.6934784215662
                                        ],
                                        [
                                            46168,
                                            2046.066304507918
                                        ],
                                        [
                                            46169,
                                            2046.4391305942698
                                        ],
                                        [
                                            46170,
                                            2046.8119566806216
                                        ],
                                        [
                                            46171,
                                            2047.1847827669738
                                        ],
                                        [
                                            46172,
                                            2047.557608853326
                                        ],
                                        [
                                            46173,
                                            2047.9304349396778
                                        ],
                                        [
                                            46174,
                                            2048.3032610260299
                                        ],
                                        [
                                            46175,
                                            2048.6760871123817
                                        ],
                                        [
                                            46176,
                                            2049.0489131987335
                                        ],
                                        [
                                            46177,
                                            2049.4217392850857
                                        ],
                                        [
                                            46178,
                                            2049.7945653714375
                                        ],
                                        [
                                            46179,
                                            2050.167391457789
                                        ],
                                        [
                                            46180,
                                            2050.540217544141
                                        ],
                                        [
                                            46181,
                                            2050.913043630493
                                        ],
                                        [
                                            46182,
                                            2051.285869716845
                                        ],
                                        [
                                            46183,
                                            2051.6586958031969
                                        ],
                                        [
                                            46184,
                                            2052.0315218895486
                                        ],
                                        [
                                            46185,
                                            2052.4043479759009
                                        ],
                                        [
                                            46186,
                                            2052.7771740622526
                                        ],
                                        [
                                            46187,
                                            2053.1500001486044
                                        ],
                                        [
                                            46188,
                                            2053.5228262349567
                                        ],
                                        [
                                            46189,
                                            2053.8956523213084
                                        ],
                                        [
                                            46190,
                                            2054.26847840766
                                        ],
                                        [
                                            46191,
                                            2054.6413044940125
                                        ],
                                        [
                                            46192,
                                            2055.014130580364
                                        ],
                                        [
                                            46193,
                                            2055.386956666716
                                        ],
                                        [
                                            46194,
                                            2055.7597827530678
                                        ],
                                        [
                                            46195,
                                            2056.13260883942
                                        ],
                                        [
                                            46196,
                                            2056.5054349257718
                                        ],
                                        [
                                            46197,
                                            2056.8782610121236
                                        ],
                                        [
                                            46198,
                                            2057.251087098475
                                        ],
                                        [
                                            46199,
                                            2057.6239131848276
                                        ],
                                        [
                                            46200,
                                            2057.996739271179
                                        ],
                                        [
                                            46201,
                                            2058.3695653575316
                                        ],
                                        [
                                            46202,
                                            2058.742391443883
                                        ],
                                        [
                                            46203,
                                            2059.115217530235
                                        ],
                                        [
                                            46204,
                                            2059.488043616587
                                        ],
                                        [
                                            46205,
                                            2059.860869702939
                                        ],
                                        [
                                            46206,
                                            2060.233695789291
                                        ],
                                        [
                                            46207,
                                            2060.6065218756427
                                        ],
                                        [
                                            46208,
                                            2060.979347961995
                                        ],
                                        [
                                            46209,
                                            2061.3521740483467
                                        ],
                                        [
                                            46210,
                                            2061.725000134699
                                        ],
                                        [
                                            46211,
                                            2062.0978262210508
                                        ],
                                        [
                                            46212,
                                            2062.4706523074025
                                        ],
                                        [
                                            46213,
                                            2062.8434783937548
                                        ],
                                        [
                                            46214,
                                            2063.2163044801066
                                        ],
                                        [
                                            46215,
                                            2063.589130566458
                                        ],
                                        [
                                            46216,
                                            2063.96195665281
                                        ],
                                        [
                                            46217,
                                            2064.334782739162
                                        ],
                                        [
                                            46218,
                                            2064.707608825514
                                        ],
                                        [
                                            46219,
                                            2065.0804349118659
                                        ],
                                        [
                                            46220,
                                            2065.453260998218
                                        ],
                                        [
                                            46221,
                                            2065.8260870845699
                                        ],
                                        [
                                            46222,
                                            2066.1989131709217
                                        ],
                                        [
                                            46223,
                                            2066.5717392572735
                                        ],
                                        [
                                            46224,
                                            2066.9445653436257
                                        ],
                                        [
                                            46225,
                                            2067.3173914299775
                                        ],
                                        [
                                            46226,
                                            2067.6902175163297
                                        ],
                                        [
                                            46227,
                                            2068.0630436026815
                                        ],
                                        [
                                            46228,
                                            2068.435869689033
                                        ],
                                        [
                                            46229,
                                            2068.808695775385
                                        ],
                                        [
                                            46230,
                                            2069.181521861737
                                        ],
                                        [
                                            46231,
                                            2069.554347948089
                                        ],
                                        [
                                            46232,
                                            2069.927174034441
                                        ],
                                        [
                                            46233,
                                            2070.300000120793
                                        ],
                                        [
                                            46234,
                                            2070.493406711503
                                        ],
                                        [
                                            46235,
                                            2070.6868133022137
                                        ],
                                        [
                                            46236,
                                            2070.880219892924
                                        ],
                                        [
                                            46237,
                                            2071.073626483634
                                        ],
                                        [
                                            46238,
                                            2071.2670330743446
                                        ],
                                        [
                                            46239,
                                            2071.4604396650549
                                        ],
                                        [
                                            46240,
                                            2071.653846255765
                                        ],
                                        [
                                            46241,
                                            2071.8472528464756
                                        ],
                                        [
                                            46242,
                                            2072.0406594371859
                                        ],
                                        [
                                            46243,
                                            2072.2340660278966
                                        ],
                                        [
                                            46244,
                                            2072.4274726186069
                                        ],
                                        [
                                            46245,
                                            2072.6208792093168
                                        ],
                                        [
                                            46246,
                                            2072.8142858000276
                                        ],
                                        [
                                            46247,
                                            2073.0076923907379
                                        ],
                                        [
                                            46248,
                                            2073.201098981448
                                        ],
                                        [
                                            46249,
                                            2073.394505572158
                                        ],
                                        [
                                            46250,
                                            2073.5879121628688
                                        ],
                                        [
                                            46251,
                                            2073.781318753579
                                        ],
                                        [
                                            46252,
                                            2073.9747253442894
                                        ],
                                        [
                                            46253,
                                            2074.168131934999
                                        ],
                                        [
                                            46254,
                                            2074.36153852571
                                        ],
                                        [
                                            46255,
                                            2074.5549451164208
                                        ],
                                        [
                                            46256,
                                            2074.7483517071307
                                        ],
                                        [
                                            46257,
                                            2074.941758297841
                                        ],
                                        [
                                            46258,
                                            2075.1351648885517
                                        ],
                                        [
                                            46259,
                                            2075.328571479262
                                        ],
                                        [
                                            46260,
                                            2075.5219780699719
                                        ],
                                        [
                                            46261,
                                            2075.715384660682
                                        ],
                                        [
                                            46262,
                                            2075.908791251393
                                        ],
                                        [
                                            46263,
                                            2076.1021978421028
                                        ],
                                        [
                                            46264,
                                            2076.295604432813
                                        ],
                                        [
                                            46265,
                                            2076.4890110235239
                                        ],
                                        [
                                            46266,
                                            2076.682417614234
                                        ],
                                        [
                                            46267,
                                            2076.875824204944
                                        ],
                                        [
                                            46268,
                                            2077.069230795654
                                        ],
                                        [
                                            46269,
                                            2077.262637386365
                                        ],
                                        [
                                            46270,
                                            2077.4560439770755
                                        ],
                                        [
                                            46271,
                                            2077.649450567785
                                        ],
                                        [
                                            46272,
                                            2077.842857158496
                                        ],
                                        [
                                            46273,
                                            2078.0362637492064
                                        ],
                                        [
                                            46274,
                                            2078.229670339916
                                        ],
                                        [
                                            46275,
                                            2078.423076930627
                                        ],
                                        [
                                            46276,
                                            2078.616483521337
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

    cx.load_json((json, 'hedge_test.json'))
    calc, result = cx.run_job()

    artifact_path = save_policy_artifact(
        result.policy_artifact,
        os.path.join('artifacts', 'policy_test_policy_artifact.json'),
    )

    print(result.evaluation_summary)
    print({'policy_artifact_path': str(artifact_path)})