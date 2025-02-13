import subprocess
import os.path


class Install:
    f5TTSPath = os.path.join(os.path.dirname(__file__), "F5-TTS")

    @staticmethod
    def check_install():
        if not os.path.exists(os.path.join(Install.f5TTSPath, "README.md")):
            Install.install()

    @staticmethod
    def install():
        print("F5TTS. Checking out submodules")
        subprocess.run(
            ['git', 'submodule', 'update', '--init', '--recursive'],
            cwd=os.path.dirname(__file__),
            shell=True,
            )

    @staticmethod
    def clone():
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/SWivid/F5-TTS",
                "F5-TTS"
            ],
            cwd=os.path.dirname(__file__)
            )
