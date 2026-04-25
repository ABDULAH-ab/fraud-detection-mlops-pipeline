from kfp import compiler

from pipeline import fraud_pipeline


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=fraud_pipeline,
        package_path="fraud_pipeline.yaml",
    )
