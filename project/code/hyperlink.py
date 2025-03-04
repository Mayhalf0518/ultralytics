from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time

# 設定 WebDriver（假設你用的是 Chrome）
driver = webdriver.Chrome()
driver.get("http://10.22.54.143:4747")  # DroidCamX 網頁

time.sleep(2)  # 等待頁面加載

try:
    # 找到 <a href="/override"> 這個元素
    override_link = driver.find_element(By.XPATH, "//a[@href='/override']")
    override_link.click()  # 自動點擊
    print("✅ 成功點擊 Override 連結，準備讀取影像")

    time.sleep(2)  # 等待影像載入

    # 檢查影像是否成功載入
    img_element = driver.find_element(By.TAG_NAME, "img")  # 嘗試抓取影像
    print("✅ 找到影像播放區域")

except Exception as e:
    print(f"❌ 無法點擊 Override 連結: {e}")

# 注意：如果影像已經出現，可以用 OpenCV 來擷取影像（可參考之前的 OpenCV 方案）
