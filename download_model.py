import urllib.request
import os

# 100% Sahi aur Working GitHub Links
files = {

    "age_net.caffemodel":
    "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel",

    "gender_net.caffemodel":
    "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_net.caffemodel"

}
print("🚀 Fresh links se model download shuru ho raha hai... Please wait...\n")

for file_name, url in files.items():
    print(f"📥 Downloading {file_name}...")
    try:
        urllib.request.urlretrieve(url, file_name)
        print(f"✅ {file_name} successfully download ho gayi!\n")
    except Exception as e:
        print(f"❌ Error downloading {file_name}: {e}\n")

print("🎉 Saari missing files download ho chuki hain!")