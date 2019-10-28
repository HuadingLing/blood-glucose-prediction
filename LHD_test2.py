from datasets import ohio

x_train, y_train, x_valid, y_valid, x_test, y_test = ohio.load_glucose_dataset(
        xlsx_path='./data/unprocessed_cgm_data.xlsx',
                         nb_past_steps=6, 
                         nb_future_steps=6,
                         train_fraction=0.6,
                         valid_fraction=0.2,
                         test_fraction=0.2, 
                         sheet_name='Baseline', 
                         patient_id=2, 
                         data_start_pos=7, 
                         max_length=30)