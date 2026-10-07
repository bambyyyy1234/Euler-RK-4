# Interactive Slope Field & Euler/RK4 ODE Solver

เว็บแอปแสดง Slope Field ของ dy/dx = f(x, y) และเปรียบเทียบวิธีออยเลอร์กับรุงเงอ-คุตตาอันดับ 4

## ไฟล์
- `app.py` — หน้าเว็บ (Streamlit)
- `ode_core.py` — ส่วนคำนวณ (parser, Euler, RK4, จุดสมดุล, ตรวจเงื่อนไขทฤษฎีบท, CoordMap)
- `requirements.txt` — ไลบรารีที่ต้องติดตั้ง

## รันในเครื่อง (Windows cmd)
```
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## นำขึ้นออนไลน์ (Streamlit Community Cloud)
1. สร้าง repository บน GitHub แล้ว push ไฟล์ `app.py`, `ode_core.py`, `requirements.txt`, `.gitignore`
   (ห้าม push โฟลเดอร์ `.venv`)
   ```
   git init
   git add app.py ode_core.py requirements.txt README.md .gitignore
   git commit -m "Slope field app"
   git branch -M main
   git remote add origin https://github.com/<ชื่อผู้ใช้>/<ชื่อ repo>.git
   git push -u origin main
   ```
2. เข้า https://share.streamlit.io แล้วล็อกอินด้วย GitHub
3. กด Create app → เลือก repository, branch `main`, Main file path = `app.py`
4. ใน Advanced settings เลือกเวอร์ชัน Python ที่ระบบรองรับ (ถ้าไม่มี 3.14 ให้ใช้ 3.12 หรือ 3.13)
5. กด Deploy แล้วจะได้ลิงก์สำหรับเปิดบนเบราว์เซอร์
