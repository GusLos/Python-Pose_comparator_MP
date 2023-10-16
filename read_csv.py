import pandas

class ReadCSV():

    @classmethod
    def read_pose_data(cls, csv_file_path: str) -> dict:
        '''
        Recebe uma string, path do arquivo .csv

        Retorna o dicionario ("JSON") com informações da pose no formato usado/gerado pelo PoseComparator:


       
        '''
        df = pandas.read_csv(csv_file_path, delimiter=';')

        result = {
            'upper_limbs' : {
                'right' : {
                    'arm': {
                        'angle'         : df.mean()['right_arm_angle'],
                        'angle_std'     : df.std()['right_arm_angle'],
                        'direction'     : df.mean()['right_arm_direction'],
                        'direction_std' : df.std()['right_arm_direction'],
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
                        'direction'     : df.mean()['left_arm_direction'],
                        'direction_std' : df.std()['left_arm_direction'],
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
    print(result['shoulders'])
    
    pass