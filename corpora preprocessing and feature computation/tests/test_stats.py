from graded_readers_stats.stats import collect_stats_keys


def test_stats():
    # Given
    input = [
        {  # doc 1
            "upos": {
                "count": 658,
                "vals": {
                    "ADJ": {
                        "count": 55,
                        "vals": {
                            "Degree": {
                                "count": 12,
                                "vals": {
                                    "Cmp": 5
                                }
                            }
                        }
                    },
                    "ADP": {
                        "count": 78,
                        "vals": {
                            "ADP-1": {
                                "count": 1,
                                "vals": {
                                    "blah1": 0
                                }
                            }
                        }
                    }
                }
            },
        },
        {  # doc 2
            "deprel": {
                "acl": 123,
                "advcl": 456
            },
            "upos": {
                "count": 666,
                "vals": {
                    "ADP": {
                        "count": 55,
                        "vals": {
                            "ADP-2": {
                                "count": 1,
                                "vals": {
                                    "blah2": 0
                                }
                            }
                        }
                    },
                    "ADV": {
                        "count": 98,
                        "vals": {
                            "Polarity": {
                                "count": 44,
                                "vals": {
                                    "Neg": 24
                                }
                            }
                        }
                    },
                    "AUX": {
                        "count": 78,
                        "vals": {
                            "Mood": {
                                "count": 8,
                                "vals": {
                                    "Ind": 8
                                }
                            }
                        }
                    }
                }
            }
        }
    ]

    # When
    result = {}
    for d in input:
        collect_stats_keys(result, d, [])

    # Then
    assert {
               "upos": {
                   "count": 0,
                   "vals": {
                       "ADJ": {
                           "count": 0,
                           "vals": {
                               "Degree": {
                                   "count": 0,
                                   "vals": {
                                       "Cmp": 0
                                   }
                               }
                           }
                       },
                       "ADP": {
                           "count": 0,
                           "vals": {
                               "ADP-1": {
                                   "count": 0,
                                   "vals": {
                                       "blah1": 0
                                   }
                               },
                               "ADP-2": {
                                   "count": 0,
                                   "vals": {
                                       "blah2": 0
                                   }
                               }
                           }
                       },
                       "ADV": {
                           "count": 0,
                           "vals": {
                               "Polarity": {
                                   "count": 0,
                                   "vals": {
                                       "Neg": 0
                                   }
                               }
                           }
                       },
                       "AUX": {
                           "count": 0,
                           "vals": {
                               "Mood": {
                                   "count": 0,
                                   "vals": {
                                       "Ind": 0
                                   }
                               }
                           }
                       }
                   }
               },
               "deprel": {
                   "acl": 0,
                   "advcl": 0
               }
           } == result


