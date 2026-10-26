import zipfile, pathlib
p = pathlib.Path('app_bundle.zip')
with zipfile.ZipFile(p) as z:
    names = z.namelist()
    print('app.py' in names)
    data = z.read('app.py').decode('utf-8', errors='replace')
    print('Live View' in data)
    print(data.count('def render_live_data_page'))
    idx = data.find('pages = ["Dashboard", "My Profile"]')
    print(idx)
    if idx != -1:
        print(data[idx:idx+500])
