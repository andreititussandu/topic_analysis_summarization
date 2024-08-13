import pandas as pd

def extract_training_data(input_file, output_file):
    try:
        chunk_size = 10000
        chunks = pd.read_csv(input_file, delimiter=';', chunksize=chunk_size, on_bad_lines='skip')

        topics = ['SCIENCE', 'TECHNOLOGY', 'BUSINESS', 'HEALTH', 'WORLD', 'ENTERTAINMENT', 'SPORTS']
        topic_data = {topic: [] for topic in topics}

        for chunk in chunks:
            for topic in topics:
                topic_rows = chunk[chunk['topic'] == topic]
                topic_data[topic].extend(topic_rows.to_dict(orient='records'))
                if len(topic_data[topic]) >= 1000:
                    topic_data[topic] = topic_data[topic][:1000]

            if all(len(data) >= 1000 for data in topic_data.values()):
                break

        combined_data = []
        for topic in topics:
            combined_data.extend(topic_data[topic][:1000])

        combined_df = pd.DataFrame(combined_data)
        combined_df.to_csv(output_file, index=False)

        print(f"Training data extracted and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    input_file = '../data/labelled_newscatcher_dataset_test.csv'
    output_file = '../data/training_data.csv'
    extract_training_data(input_file, output_file)
