from pipe import ClassifierPipeline
import config as config

if __name__ == "__main__":
    pipeline = ClassifierPipeline(config)
    pipeline.run_random_search()