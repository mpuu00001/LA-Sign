mt5_path = "./pretrained_weight/mt5-base"
mt5_aux_path = "./pretrained_weight/mt5-base-trim"

# label paths
train_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "OpenASL": "./data/OpenASL/labels.train",
                    "How2sign": "./data/How2Sign/labels.train"
                    }

dev_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev",
                    "OpenASL": "./data/OpenASL/labels.val"
                    }

test_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test",
                    "OpenASL": "./data/OpenASL/labels.test",
                    "How2sign": "./data/How2Sign/labels.test"
                    }

# pose paths
pose_dirs = {
            "CSL_News": './dataset/CSL_News/pose_format',
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format",
            "OpenASL": "./dataset/OpenASL/pose_format",
            "How2sign": "./dataset/How2sign/pose_format"
            }

pose_dirs_gpu = {            
            "CSL_News": '/CSL_News/pose_format',
            "CSL_Daily": '/CSL_Daily/pose_format',
            "WLASL": "/WLASL/pose_format",
            "OpenASL": "/OpenASL/pose_format",
            "How2sign": "/How2sign/pose_format"
            }

# video paths
rgb_dirs = {
            "CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format",
            "OpenASL": './dataset/WLASL/rgb_format',
            "How2sign": "./dataset/How2sign/rgb_format"
            }

rgb_dirs_gpu = {
            "CSL_News": '/CSL_News/rgb_format',
            "CSL_Daily": '/CSL_Daily/sentence-crop',
            "WLASL": '/WLASL/rgb_format',
            "OpenASL": '/OpenASL/rgb_format',
            "How2sign": '/How2sign/rgb_format'
            }
