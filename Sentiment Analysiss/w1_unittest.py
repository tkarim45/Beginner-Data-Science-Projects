# import nltk
import numpy as np


def test_sigmoid(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {"name": "default_check", "input": {"z": 0}, "expected": 0.5},
        {
            "name": "positive_check",
            "input": {"z": 4.92},
            "expected": 0.9927537604041685,
        },
        {"name": "negative_check", "input": {"z": -1}, "expected": 0.2689414213699951},
        {
            "name": "larger_neg_check",
            "input": {"z": -20},
            "expected": 2.0611536181902037e-09,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output from sigmoid function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_gradientDescent(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "random_seed": 1,
                "input_dict": {
                    "x": np.array(
                        [
                            [1.00000000e00, 8.34044009e02, 1.44064899e03],
                            [1.00000000e00, 2.28749635e-01, 6.04665145e02],
                            [1.00000000e00, 2.93511782e02, 1.84677190e02],
                            [1.00000000e00, 3.72520423e02, 6.91121454e02],
                            [1.00000000e00, 7.93534948e02, 1.07763347e03],
                            [1.00000000e00, 8.38389029e02, 1.37043900e03],
                            [1.00000000e00, 4.08904499e02, 1.75623487e03],
                            [1.00000000e00, 5.47751864e01, 1.34093502e03],
                            [1.00000000e00, 8.34609605e02, 1.11737966e03],
                            [1.00000000e00, 2.80773877e02, 3.96202978e02],
                        ]
                    ),  
                    "y": np.array(
                        [
                            [1.0],
                            [1.0],
                            [0.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [1.0],
                        ]
                    ),  
                    "theta": np.zeros((3, 1)),
                    "alpha": 1e-8,
                    "num_iters": 700,
                },
            },
            "expected": {
                "J": 0.6709497038162118,
                "theta": np.array(
                    [[4.10713435e-07], [3.56584699e-04], [7.30888526e-05]]
                ),
            },
        },
        {
            "name": "larger_check",
            "input": {
                "random_seed": 2,
                "input_dict": {
                    "x": np.array(
                        [
                            [1.0, 435.99490214, 25.92623183, 549.66247788],
                            [1.0, 435.32239262, 420.36780209, 330.334821],
                            [1.0, 204.64863404, 619.27096635, 299.65467367],
                            [1.0, 266.8272751, 621.13383277, 529.14209428],
                            [1.0, 134.57994534, 513.57812127, 184.43986565],
                            [1.0, 785.33514782, 853.97529264, 494.23683738],
                            [1.0, 846.56148536, 79.64547701, 505.24609012],
                            [1.0, 65.28650439, 428.1223276, 96.53091566],
                            [1.0, 127.1599717, 596.74530898, 226.0120006],
                            [1.0, 106.94568431, 220.30620707, 349.826285],
                            [1.0, 467.78748458, 201.74322626, 640.40672521],
                            [1.0, 483.06983555, 505.23672002, 386.89265112],
                            [1.0, 793.63745444, 580.00417888, 162.2985985],
                            [1.0, 700.75234661, 964.55108009, 500.00836117],
                            [1.0, 889.52006395, 341.61365267, 567.14412763],
                            [1.0, 427.5459633, 436.74726303, 776.559185],
                            [1.0, 535.6041735, 953.74222694, 544.20816015],
                            [1.0, 82.09492228, 366.34240168, 850.850504],
                            [1.0, 406.27504305, 27.20236589, 247.177239],
                            [1.0, 67.14437074, 993.85201142, 970.58031338],
                        ]
                    ),  
                    "y": np.array(
                        [
                            [1.0],
                            [1.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [1.0],
                            [0.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [1.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [1.0],
                            [0.0],
                        ]
                    ),  
                    "theta": np.zeros((4, 1)),
                    "alpha": 1e-4,
                    "num_iters": 30,
                },
            },
            "expected": {
                "J": 6.5044107216556135,
                "theta": np.array(
                    [
                        [9.45211976e-05],
                        [2.40577958e-02],
                        [-1.77876847e-02],
                        [1.35674845e-02],
                    ]
                ),
            },
        },
    ]

    for test_case in test_cases:
        # Setting the random seed for reproducibility
        result_J, result_theta = target(**test_case["input"]["input_dict"])

        try:
            assert isinstance(result_J, float)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": float, "got": type(result_J),}
            )
            print(
                f"Wrong output type for loss function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result_J, test_case["expected"]["J"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["J"],
                    "got": result_J,
                }
            )
            print(
                f"Wrong output for the loss function. Check how you are implementing the matrix multiplications. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_theta.shape == test_case["input"]["input_dict"]["theta"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["input"]["input_dict"]["theta"].shape,
                    "got": result_theta.shape,
                }
            )
            print(
                f"Wrong shape for weights matrix theta. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                np.squeeze(result_theta), np.squeeze(test_case["expected"]["theta"]),
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["theta"],
                    "got": result_theta,
                }
            )
            print(
                f"Wrong values for weight's matrix theta. Check how you are updating the matrix of weights. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
    


# +
def test_extract_features(target, freqs):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "tweet": "#FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)",
                "freqs": freqs,
            },
            "expected": np.array(
                [[1.00e00, 3.133e03, 6.10e01]]
            ), 
        },
        {
            "name": "unk_words_check",
            "input": {"tweet": "blorb bleeeeb bloooob", "freqs": freqs},
            "expected": np.array([[1.0, 0.0, 0.0]]),
        },
        {
            "name": "good_words_check",
            "input": {"tweet": "Hello world! All's good!", "freqs": freqs},
            "expected": np.array([[1.0, 263.0, 106.0]]),
        },
        {
            "name": "bad_words_check",
            "input": {"tweet": "It is so sad!", "freqs": freqs},
            "expected": np.array([[1.0, 5.0, 100.0]]),
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result.shape == test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Wrong output shape. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output values. Check how you are computing the positive or negative word count. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


# -

def test_predict_tweet(target, freqs, theta):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check1",
            "input": {"tweet": "I am happy", "freqs": freqs, "theta": theta},
            "expected": np.array([[0.5192746]]),
        },
        {
            "name": "default_check2",
            "input": {"tweet": "I am bad", "freqs": freqs, "theta": theta},
            "expected": np.array([[0.49434685]]),
        },
        {
            "name": "default_check3",
            "input": {
                "tweet": "this movie should have been great",
                "freqs": freqs,
                "theta": theta,
            },
            "expected": np.array([[0.5159792]]), 
        },
        {
            "name": "default_check5",
            "input": {"tweet": "It is a good day", "freqs": freqs, "theta": theta,},
            "expected": np.array([[0.52320595]]), 
        },
        {
            "name": "default_check6",
            "input": {"tweet": "It is a bad bad day", "freqs": freqs, "theta": theta,},
            "expected": np.array([[0.49780224]]), 
        },
        {
            "name": "default_check7",
            "input": {
                "tweet": "It is a good day",
                "freqs": freqs,
                "theta": np.array([[5.0000e-04], [-3.4e-02], [3.2e-02]]),
            },
            "expected": np.array([[0.00147813]]), 
        },
        {
            "name": "default_check8",
            "input": {
                "tweet": "It is a bad bad day",
                "freqs": freqs,
                "theta": np.array([[5.0000e-04], [-3.4e-02], [3.2e-02]]),
            },
            "expected": np.array([[0.45673348]]), 
        },
        {
            "name": "default_check9",
            "input": {
                "tweet": "this movie should have been great",
                "freqs": freqs,
                "theta": np.array([[5.0000e-04], [-3.4e-02], [3.2e-02]]),
            },
            "expected": np.array([[0.01561938]]), 
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result.shape == test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Wrong output shape. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong predicted values. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
    


def unittest_test_logistic_regression(target, freqs, theta):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check1",
            "input": {
                "test_x": [
                    "Bro:U wan cut hair anot,ur hair long Liao bo\nMe:since ord liao,take it easy lor treat as save $ leave it longer :)\nBro:LOL Sibei xialan",
                    "@heyclaireee is back! thnx God!!! i'm so happy :)",
                    "@BBCRadio3 thought it was my ears which were malfunctioning, thank goodness you cleared that one up with an apology :-)",
                    "@HumayAG 'Stuck in the centre right with you. Clowns to the right, jokers to the left...' :) @orgasticpotency @ahmedshaheed @AhmedSaeedGahaa",
                    "Happy Friday :-) http://t.co/iymPIlWXFY",
                    "I wanna change my avi but uSanele :(",
                    "MY PUPPY BROKE HER FOOT :(",
                    "where's all the jaebum baby pictures :((",
                    "But but Mr Ahmad Maslan cooks too :( https://t.co/ArCiD31Zv6",
                    "@eawoman As a Hull supporter I am expecting a misserable few weeks :-(",
                ],
                "test_y": np.array(
                    [
                        [1.0],
                        [1.0],
                        [1.0],
                        [1.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ]
                ),
                "freqs": freqs,
                "theta": theta,
            },
            "expected": 1.0,
        },
        {
            "name": "default_check1",
            "input": {
                "test_x": [
                    "Bro:U wan cut hair anot,ur hair long Liao bo\nMe:since ord liao,take it easy lor treat as save $ leave it longer :)\nBro:LOL Sibei xialan",
                    "@heyclaireee is back! thnx God!!! i'm so happy :)",
                    "@BBCRadio3 thought it was my ears which were malfunctioning, thank goodness you cleared that one up with an apology :-)",
                    "@HumayAG 'Stuck in the centre right with you. Clowns to the right, jokers to the left...' :) @orgasticpotency @ahmedshaheed @AhmedSaeedGahaa",
                    "Happy Friday :-) http://t.co/iymPIlWXFY",
                    "I wanna change my avi but uSanele :(",
                    "MY PUPPY BROKE HER FOOT :(",
                    "where's all the jaebum baby pictures :((",
                    "But but Mr Ahmad Maslan cooks too :( https://t.co/ArCiD31Zv6",
                    "@eawoman As a Hull supporter I am expecting a misserable few weeks :-(",
                ],
                "test_y": np.array(
                    [
                        [1.0],
                        [1.0],
                        [1.0],
                        [1.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ]
                ),
                "freqs": freqs,
                "theta": np.array([[5.0000e-04], [-3.4e-02], [3.2e-02]]),
            },
            "expected": 0.0,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, np.float64)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.float64,
                    "got": type(result),
                }
            )
            print(
                f"Wrong output type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong accuracy value. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
    
