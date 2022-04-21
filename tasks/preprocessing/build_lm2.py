import subprocess

class BuildLM:
    def __init__(self, lm_path):
        self.lm_path = lm_path

    def build_lm(self):
        subprocess.run([self.lm_path])

    def __call__(self):
        return self.build_lm()


if __name__ == "__main__":

    build = BuildLM(lm_path='./build_lm.sh')
    build()
