import pandas

class ReadCSV():

    @classmethod
    def read_pose_data(cls, csv_file_path: str) -> dict:
        '''
        Recebe uma string, path do arquivo .csv

        Retorna o dicionario ("JSON") com informações da pose no formato usado/gerado pelo PoseComparator:


        result = {
            'upper_limbs' : {
                'right' : {
                    'arm': {
                        'angle'         : right_arm_angle,
                        'angle_std'     : right_arm_angle_std
                        'director_i'    : right_arm_director_i,
                        'director_i_std': right_arm_director_i_std,
                        'director_j'    : right_arm_director_j,
                        'director_j_std': right_arm_director_j_std,
                        'director_k'    : right_arm_director_k,
                        'director_k_std': right_arm_director_k_std,
                    },
                    'forearm': {
                        'angle'         : right_forearm_angle,
                        'angle_std'     : right_forearm_angle_std
                        'director_i'    : right_forearm_director_i,
                        'director_i_std': right_forearm_director_i_std,
                        'director_j'    : right_forearm_director_j,
                        'director_j_std': right_forearm_director_j_std,
                        'director_k'    : right_forearm_director_k,
                        'director_k_std': right_forearm_director_k_std,
                    },
                },
                'left' : {
                    'arm': {
                        'angle'         : left_arm_angle,
                        'angle_std'     : left_arm_angle_std
                        'director_i'    : left_arm_director_i,
                        'director_i_std': left_arm_director_i_std,
                        'director_j'    : left_arm_director_j,
                        'director_j_std': left_arm_director_j_std,
                        'director_k'    : left_arm_director_k,
                        'director_k_std': left_arm_director_k_std,
                    },
                    'forearm': {
                        'angle'         : left_forearm_angle,
                        'angle_std'     : left_forearm_angle_std
                        'director_i'    : left_forearm_director_i,
                        'director_i_std': left_forearm_director_i_std,
                        'director_j'    : left_forearm_director_j,
                        'director_j_std': left_forearm_director_j_std,
                        'director_k'    : left_forearm_director_k,
                        'director_k_std': left_forearm_director_k_std,
                    },
                },
            },
            'shoulders': {
                director_i     : shoulders_director_i,
                director_i_std : shoulders_director_i_std,
                director_j     : shoulders_director_j,
                director_j_std : shoulders_director_j_std,
                director_k     : shoulders_director_k,
                director_k_std : shoulders_director_k_std,
            },   
            'neck': {
                director_i     : neck_director_i,
                director_i_std : neck_director_i_std,
                director_j     : neck_director_j,
                director_j_std : neck_director_j_std,
                director_k     : neck_director_k,
                director_k_std : neck_director_k_std,
            }
        }
        '''
        df = pandas.read_csv(csv_file_path, delimiter=';')

        result = {
            'upper_limbs' : {
                'right' : {
                    'arm': {
                        'angle'         : df.mean()['right_arm_angle'],
                        'angle_std'     : df.std()['right_arm_angle'],
                        # 'director_i'    : df.mean()['right_arm_director_i'],
                        # 'director_i_std': df.std()['right_arm_director_i'],
                        # 'director_j'    : df.mean()['right_arm_director_j'],
                        # 'director_j_std': df.std()['right_arm_director_j'],
                        # 'director_k'    : df.mean()['right_arm_director_k'],
                        # 'director_k_std': df.std()['right_arm_director_k'],
                        'director':{
                            0: df.mean()['right_arm_director_i'],
                            1: df.mean()['right_arm_director_j'],
                            2: df.mean()['right_arm_director_k'],
                        },
                        'director_std':{
                            0: df.std()['right_arm_director_i'],
                            1: df.std()['right_arm_director_j'],
                            2: df.std()['right_arm_director_k'],
                        },
                    },
                    'forearm': {
                        'angle'         : df.mean()['right_forearm_angle'],
                        'angle_std'     : df.std()['right_forearm_angle'],
                        # 'director_i'    : df.mean()['right_forearm_director_i'],
                        # 'director_i_std': df.std()['right_forearm_director_i'],
                        # 'director_j'    : df.mean()['right_forearm_director_j'],
                        # 'director_j_std': df.std()['right_forearm_director_j'],
                        # 'director_k'    : df.mean()['right_forearm_director_k'],
                        # 'director_k_std': df.std()['right_forearm_director_k'],
                        'director':{
                            0: df.mean()['right_forearm_director_i'],
                            1: df.mean()['right_forearm_director_j'],
                            2: df.mean()['right_forearm_director_k'],
                        },
                        'director_std':{
                            0: df.std()['right_forearm_director_i'],
                            1: df.std()['right_forearm_director_j'],
                            2: df.std()['right_forearm_director_k'],
                        },
                    },
                },
                'left' : {
                    'arm': {
                        'angle'         : df.mean()['left_arm_angle'],
                        'angle_std'     : df.std()['left_arm_angle'],
                        # 'director_i'    : df.mean()['left_arm_director_i'],
                        # 'director_i_std': df.std()['left_arm_director_i'],
                        # 'director_j'    : df.mean()['left_arm_director_j'],
                        # 'director_j_std': df.std()['left_arm_director_j'],
                        # 'director_k'    : df.mean()['left_arm_director_k'],
                        # 'director_k_std': df.std()['left_arm_director_k'],
                        'director':{
                            0: df.mean()['left_arm_director_i'],
                            1: df.mean()['left_arm_director_j'],
                            2: df.mean()['left_arm_director_k'],
                        },
                        'director_std':{
                            0: df.std()['left_arm_director_i'],
                            1: df.std()['left_arm_director_j'],
                            2: df.std()['left_arm_director_k'],
                        },
                    },
                    'forearm': {
                        'angle'         : df.mean()['left_forearm_angle'],
                        'angle_std'     : df.std()['left_forearm_angle'],
                        # 'director_i'    : df.mean()['left_forearm_director_i'],
                        # 'director_i_std': df.std()['left_forearm_director_i'],
                        # 'director_j'    : df.mean()['left_forearm_director_j'],
                        # 'director_j_std': df.std()['left_forearm_director_j'],
                        # 'director_k'    : df.mean()['left_forearm_director_k'],
                        # 'director_k_std': df.std()['left_forearm_director_k'],
                        'director':{
                            0: df.mean()['left_forearm_director_i'],
                            1: df.mean()['left_forearm_director_j'],
                            2: df.mean()['left_forearm_director_k'],
                        },
                        'director_std':{
                            0: df.std()['left_forearm_director_i'],
                            1: df.std()['left_forearm_director_j'],
                            2: df.std()['left_forearm_director_k'],
                        },
                    },
                },
            },
            'shoulders': {
                # 'director_i'     : df.mean()['shoulders_i'],
                # 'director_i_std' : df.std()['shoulders_i'],
                # 'director_j'     : df.mean()['shoulders_j'],
                # 'director_j_std' : df.std()['shoulders_j'],
                # 'director_k'     : df.mean()['shoulders_k'],
                # 'director_k_std' : df.std()['shoulders_k'],
                'director': {
                    0: df.mean()['shoulders_i'],
                    1: df.mean()['shoulders_j'],
                    2: df.mean()['shoulders_k']
                },
                'director_std': {
                    0: df.std()['shoulders_i'],
                    1: df.std()['shoulders_j'],
                    2: df.std()['shoulders_k']
                },
            },   
            'neck': {
                # 'director_i'     : df.mean()['neck_i'],
                # 'director_i_std' : df.std()['neck_i'],
                # 'director_j'     : df.mean()['neck_j'],
                # 'director_j_std' : df.std()['neck_j'],
                # 'director_k'     : df.mean()['neck_k'],
                # 'director_k_std' : df.std()['neck_k'],
                'director': {
                    0: df.mean()['neck_i'],
                    1: df.mean()['neck_j'],
                    2: df.mean()['neck_k']
                },
                'director_std': {
                    0: df.std()['neck_i'],
                    1: df.std()['neck_j'],
                    2: df.std()['neck_k']
                },
            }
        }

        return result

        pass

    pass # ReadCSV

if __name__ == "__main__":
    # df = pandas.read_csv('eggs.csv', delimiter=';')
    # print(df.columns)
    # print(30*'---')
    # # print(df['right_arm_angle'].mean())
    # print(df.describe())
    # print(30*'---')
    # # print(df.std()['right_arm_angle'])
    # print(df.mean()['right_arm_angle'])

    result = ReadCSV().read_pose_data('eggs.csv')
    print(result['upper_limbs']['right'])
    
    pass