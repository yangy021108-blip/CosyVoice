from pathlib import Path

from setuptools import setup
from torch_sdaa.utils.cpp_extension import BuildExtension, TecoExtension


ROOT = Path(__file__).resolve().parent
SDAA_HOME = Path("/opt/tecoai")

setup(
    name="sdaa_mm_encoder_fa_poc",
    ext_modules=[
        TecoExtension(
            name="sdaa_mm_encoder_fa_poc_ext",
            sources=[
                str(ROOT / "common.cc"),
                str(ROOT / "mm_encoder_fa_poc.cc"),
            ],
            include_dirs=[str(ROOT), str(SDAA_HOME / "include")],
            library_dirs=[str(SDAA_HOME / "lib64")],
            libraries=[
                "tecocustom",
                "tecolmk",
                "tecodnn",
                "tecoblas",
                "sdaart",
            ],
            extra_compile_args={
                "tecocc": ["-DUSE_EXPERIMENTAL_SDAA_API"],
                "cxx": [
                    "-Wno-unused-function",
                    "-Wno-unused-variable",
                    "-Wno-unused-but-set-variable",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
