import PyInstaller.__main__

try:

    PyInstaller.__main__.run([
        'atualiza_dados.py',
        '--onefile',
        #'--noconsole',
        #'-icamera.ico',
        '-nAtualiza Dados'
    ])
except:
    pass
