mt5_path = "./pretrained_weight/mt5-base"

# label paths
train_label_paths = {
                    "WLASL2000": "./data/WLASL/labels-2000.train",
                    "WLASL300": "./data/WLASL/labels-300.train",
                    "MSASL1000": "./data/MSASL/labels-1000.train",
                    "MSASL200": "./data/MSASL/labels-200.train",
                    "MSASL100": "./data/MSASL/labels-100.train",
                    }

dev_label_paths = {
                    "WLASL2000": "./data/WLASL/labels-2000.dev",
                    "WLASL300": "./data/WLASL/labels-300.val",
                    "MSASL1000": "./data/MSASL/labels-1000.val",
                    "MSASL200": "./data/MSASL/labels-200.val",
                    "MSASL100": "./data/MSASL/labels-100.val"
                    }

test_label_paths = {
                    "WLASL2000": "./data/WLASL/labels-2000.test",
                    "WLASL300": "./data/WLASL/labels-300.test",
                    "MSASL1000": "./data/MSASL/labels-1000.test",
                    "MSASL200": "./data/MSASL/labels-200.test",
                    "MSASL100": "./data/MSASL/labels-100.test",
                    }

# pose paths
pose_dirs = {
            "WLASL": "./dataset/WLASL/pose_format",
            "MSASL": "./dataset/MSASL/pose_format",
            }

rgb_dirs = {
            "WLASL": "./dataset/WLASL/rgb_format",
            "MSASL": "./dataset/MSASL/rgb_format",
            }
