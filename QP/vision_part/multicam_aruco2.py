import numpy as np
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import os

class ARUCOBoardPDFGenerator:
    def __init__(self) -> None:
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.size = (5, 7)  # 5x7 마커
        
        # ✅ 크기 통일 (60mm 마커, 5mm 간격)
        self.markerLength = 0.060  # 60mm
        self.markerSeparation = 0.005  # 5mm
        
        self.board = cv2.aruco.GridBoard(self.size, self.markerLength, self.markerSeparation, self.dictionary, None)
        
        # A4 크기 (210 x 297 mm)
        self.a4_width = 210
        self.a4_height = 297
        self.margin = 15  # 15mm 여백
        
        # ✅ 전체 보드 크기 계산 (정확한 공식)
        self.board_width_mm = self.size[0] * (self.markerLength * 1000) + (self.size[0] - 1) * (self.markerSeparation * 1000)
        self.board_height_mm = self.size[1] * (self.markerLength * 1000) + (self.size[1] - 1) * (self.markerSeparation * 1000)
        
        print(f"보드 전체 크기: {self.board_width_mm:.1f} x {self.board_height_mm:.1f} mm")
        print(f"마커 크기: {self.markerLength*1000:.0f}mm")
        print(f"마커 간격: {self.markerSeparation*1000:.0f}mm")
        
        # A4 페이지 분할 계산
        self.calculate_page_division()
        
    def calculate_page_division(self):
        """A4 페이지 분할 방법 계산"""
        # 사용 가능한 A4 영역 (여백과 텍스트 공간 제외)
        usable_width = self.a4_width - (2 * self.margin)   # 180mm
        usable_height = self.a4_height - (2 * self.margin) - 30  # 252mm (텍스트 30mm 확보)
        
        # 필요한 페이지 수 계산
        self.pages_x = int(np.ceil(self.board_width_mm / usable_width))
        self.pages_y = int(np.ceil(self.board_height_mm / usable_height))
        self.total_pages = self.pages_x * self.pages_y
        
        # ✅ 각 페이지의 실제 크기 미리 계산
        self.page_sizes = []
        
        for page_y in range(self.pages_y):
            for page_x in range(self.pages_x):
                # ✅ 균등 분할: 모든 페이지가 동일한 크기
                if self.pages_x == 1:
                    page_width = self.board_width_mm
                else:
                    page_width = self.board_width_mm / self.pages_x
                    
                if self.pages_y == 1:
                    page_height = self.board_height_mm  
                else:
                    page_height = self.board_height_mm / self.pages_y
                
                self.page_sizes.append({
                    'width': page_width,
                    'height': page_height,
                    'start_x': page_x * page_width,
                    'start_y': page_y * page_height
                })
        
        print(f"필요한 A4 페이지: {self.pages_x} x {self.pages_y} = {self.total_pages}장")
        print(f"A4 사용 가능 영역: {usable_width} x {usable_height} mm")
        print(f"각 페이지 크기: {page_width:.1f} x {page_height:.1f} mm")
        
    def generate_multi_page_pdf(self, filename="aruco_board_60mm.pdf"):
        """여러 페이지 PDF 생성"""
        
        # ✅ 정확한 크기를 위한 DPI 설정
        target_dpi = 300  # 고품질 DPI
        
        # ✅ 실제 물리적 크기를 픽셀로 정확히 변환
        board_width_px = int(self.board_width_mm * target_dpi / 25.4)
        board_height_px = int(self.board_height_mm * target_dpi / 25.4)
        
        print(f"목표 이미지 크기: {board_width_px} x {board_height_px} pixels @ {target_dpi}DPI")
        print(f"예상 물리 크기: {board_width_px * 25.4 / target_dpi:.1f} x {board_height_px * 25.4 / target_dpi:.1f} mm")
        
        # ✅ 전체 보드 이미지 생성 (최소 마진)
        full_board_image = self.board.generateImage(
            outSize=(board_width_px, board_height_px), 
            marginSize=5,  # 최소 마진
            borderBits=1
        )
        
        print(f"실제 생성된 이미지: {full_board_image.shape[1]} x {full_board_image.shape[0]} pixels")
        
        # ✅ 크기 검증
        self.verify_marker_size(full_board_image, target_dpi)
        
        # PDF 생성
        pdf_canvas = canvas.Canvas(filename, pagesize=A4)
        
        page_num = 1
        
        for page_y in range(self.pages_y):
            for page_x in range(self.pages_x):
                
                # ✅ 미리 계산된 페이지 크기 사용
                page_info = self.page_sizes[page_num - 1]
                
                start_x_mm = page_info['start_x']
                start_y_mm = page_info['start_y']
                actual_width_mm = page_info['width']
                actual_height_mm = page_info['height']
                
                print(f"Page {page_num}: {actual_width_mm:.1f} x {actual_height_mm:.1f} mm (균등분할)")
                
                # mm를 픽셀로 정확히 변환
                start_x_px = int(start_x_mm * target_dpi / 25.4)
                start_y_px = int(start_y_mm * target_dpi / 25.4)
                end_x_px = int((start_x_mm + actual_width_mm) * target_dpi / 25.4)
                end_y_px = int((start_y_mm + actual_height_mm) * target_dpi / 25.4)
                
                # 이미지 범위 안전 제한
                start_x_px = max(0, min(start_x_px, full_board_image.shape[1]))
                start_y_px = max(0, min(start_y_px, full_board_image.shape[0]))
                end_x_px = max(start_x_px + 1, min(end_x_px, full_board_image.shape[1]))
                end_y_px = max(start_y_px + 1, min(end_y_px, full_board_image.shape[0]))
                
                # 이미지 자르기
                cropped_image = full_board_image[start_y_px:end_y_px, start_x_px:end_x_px]
                
                if cropped_image.size == 0:
                    print(f"⚠️ Page {page_num}: 빈 이미지 건너뜀")
                    page_num += 1
                    continue
                
                # 임시 이미지 파일 저장 (고품질)
                temp_filename = f"temp_page_{page_num}.png"
                cv2.imwrite(temp_filename, cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
                # PDF에 페이지 추가 (정확한 크기로)
                self.add_page_to_pdf(pdf_canvas, temp_filename, page_x, page_y, page_num, 
                                actual_width_mm, actual_height_mm, target_dpi)
                
                # 임시 파일 삭제
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                page_num += 1
        
        pdf_canvas.save()
        print(f"✅ PDF 생성 완료: {filename}")
        
        # 조립 가이드 생성
        self.generate_assembly_guide(filename.replace('.pdf', '_assembly_guide.pdf'))
        
    def verify_marker_size(self, image, dpi):
        """생성된 이미지에서 마커 크기 검증"""
        try:
            # ArUco 마커 감지로 실제 크기 확인
            detector = cv2.aruco.ArucoDetector(self.dictionary)
            corners, ids, _ = detector.detectMarkers(image)
            
            if len(corners) > 0:
                # 첫 번째 마커의 픽셀 크기 계산
                corner = corners[0][0]
                width_px = np.linalg.norm(corner[1] - corner[0])
                height_px = np.linalg.norm(corner[2] - corner[1])
                
                # 픽셀을 mm로 변환
                width_mm = width_px * 25.4 / dpi
                height_mm = height_px * 25.4 / dpi
                
                target_size = self.markerLength * 1000
                
                print(f"🔍 마커 크기 검증:")
                print(f"   목표 크기: {target_size:.0f}mm x {target_size:.0f}mm")
                print(f"   실제 크기: {width_mm:.1f}mm x {height_mm:.1f}mm")
                print(f"   오차: {abs(width_mm - target_size):.1f}mm")
                
                if abs(width_mm - target_size) > 1.0:
                    print(f"⚠️ 크기 오차가 큽니다! DPI 또는 스케일링 문제 가능성")
                else:
                    print(f"✅ 마커 크기 정확함")
            else:
                print(f"⚠️ 마커를 감지할 수 없어 크기 검증 불가")
                
        except Exception as e:
            print(f"⚠️ 크기 검증 실패: {e}")
        
    def add_page_to_pdf(self, pdf_canvas, image_filename, page_x, page_y, page_num, 
                       actual_width_mm, actual_height_mm, source_dpi):
        """PDF에 페이지 추가 (정확한 크기 유지)"""
        
        # 페이지 정보
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(20, A4[1] - 25, f"ArUco Board - Page {page_num}/{self.total_pages}")
        
        pdf_canvas.setFont("Helvetica", 10)
        pdf_canvas.drawString(20, A4[1] - 40, f"Position: Row {page_y + 1}, Column {page_x + 1}")
        pdf_canvas.drawString(20, A4[1] - 55, f"Marker: {self.markerLength*1000:.0f}mm, Gap: {self.markerSeparation*1000:.0f}mm")
        pdf_canvas.drawString(20, A4[1] - 70, f"Page size: {actual_width_mm:.1f} x {actual_height_mm:.1f} mm")
        pdf_canvas.drawString(20, A4[1] - 85, f"PRINT at 100% - NO SCALING!")
        
        # ✅ 정확한 1:1 크기로 이미지 배치
        # ReportLab의 mm 단위를 직접 사용
        img_width_points = actual_width_mm * mm
        img_height_points = actual_height_mm * mm
        
        # 페이지 중앙 정렬
        x_pos = (A4[0] - img_width_points) / 2
        y_pos = 25 * mm  # 하단에서 25mm 위
        
        # ✅ 페이지 크기 제한 (필요시에만 스케일링)
        max_width = (self.a4_width - 2 * self.margin) * mm
        max_height = (self.a4_height - 2 * self.margin - 35) * mm
        
        scale_factor = 1.0
        if img_width_points > max_width:
            scale_factor = min(scale_factor, max_width / img_width_points)
        if img_height_points > max_height:
            scale_factor = min(scale_factor, max_height / img_height_points)
            
        if scale_factor < 1.0:
            print(f"⚠️ Page {page_num}: 크기 조정 필요 (스케일: {scale_factor:.3f})")
            img_width_points *= scale_factor
            img_height_points *= scale_factor
            x_pos = (A4[0] - img_width_points) / 2
        
        # 이미지 삽입
        try:
            pdf_canvas.drawImage(
                image_filename,
                x_pos, y_pos,
                width=img_width_points,
                height=img_height_points,
                preserveAspectRatio=True,
                anchor='c'  # 중앙 기준
            )
            print(f"✅ Page {page_num}: 이미지 삽입 성공 ({actual_width_mm:.1f}x{actual_height_mm:.1f}mm)")
        except Exception as e:
            print(f"❌ Page {page_num} 이미지 삽입 실패: {e}")
        
        # 자르기 가이드
        self.draw_cutting_guides(pdf_canvas)
        
        # 다음 페이지
        pdf_canvas.showPage()
    
    def draw_cutting_guides(self, pdf_canvas):
        """자르기 가이드 라인"""
        pdf_canvas.setStrokeColorRGB(1, 0, 0)  # 빨간색
        pdf_canvas.setLineWidth(0.5)
        
        margin = self.margin * mm
        guide_length = 5
        
        # 네 모서리에 십자 표시 (더 정확한 위치)
        corners = [
            (margin, A4[1] - margin),           # 좌상단
            (A4[0] - margin, A4[1] - margin),  # 우상단
            (margin, margin),                   # 좌하단
            (A4[0] - margin, margin)           # 우하단
        ]
        
        for x, y in corners:
            # 십자 그리기
            pdf_canvas.line(x - guide_length, y, x + guide_length, y)
            pdf_canvas.line(x, y - guide_length, x, y + guide_length)
        
        # 가이드 설명
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.setFillColorRGB(1, 0, 0)
        pdf_canvas.drawString(margin, 10, "Cut along red crosses. Align with adjacent pages.")
    
    def generate_assembly_guide(self, filename):
        """조립 가이드 생성"""
        pdf_canvas = canvas.Canvas(filename, pagesize=A4)
        
        pdf_canvas.setFont("Helvetica-Bold", 20)
        pdf_canvas.drawString(50, A4[1] - 50, "ArUco Board Assembly Guide")
        
        pdf_canvas.setFont("Helvetica", 12)
        y_pos = A4[1] - 100
        
        instructions = [
            f"전체 보드 크기: {self.board_width_mm:.1f} x {self.board_height_mm:.1f} mm",
            f"마커 크기: {self.markerLength*1000:.0f} mm (측정으로 확인 필수!)",
            f"마커 간격: {self.markerSeparation*1000:.0f} mm",
            f"총 {self.total_pages}페이지 ({self.pages_x} x {self.pages_y} 배열)",
            "",
            "🖨️ 인쇄 설정 (매우 중요!):",
            "  ✅ 크기: '실제 크기' 또는 '100% 크기' 선택",
            "  ❌ '페이지에 맞춤' 절대 사용 금지",
            "  ✅ 품질: 최고 품질 (600 DPI 이상)",
            "  ✅ 용지: A4 고급 용지",
            "  ✅ 여백: 최소 여백 설정",
            "",
            "✂️ 조립 순서:",
            "  1. 모든 페이지를 위 설정으로 인쇄",
            "  2. 자로 첫 번째 마커 측정 → 60mm 확인",
            "  3. 빨간 십자 가이드를 기준으로 여백 자르기",
            "  4. 아래 다이어그램 순서로 페이지 붙이기",
            "  5. 뒷면에서 투명테이프로 고정",
            "",
            "✅ 최종 검증:",
            f"  • 완성 후 아무 마커나 측정 → {self.markerLength*1000:.0f}mm 확인",
            "  • 마커들이 격자로 정렬되었는지 확인",
            "  • 접착 부분이 평평한지 확인",
        ]
        
        for instruction in instructions:
            pdf_canvas.drawString(50, y_pos, instruction)
            y_pos -= 16
        
        # 페이지 배치 다이어그램
        y_pos -= 30
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(50, y_pos, "📐 페이지 배치 다이어그램:")
        y_pos -= 40
        
        # 다이어그램 그리기
        start_x, start_y = 100, y_pos
        box_w, box_h = 60, 40
        
        page_num = 1
        for row in range(self.pages_y):
            for col in range(self.pages_x):
                x = start_x + col * (box_w + 10)
                y_box = start_y - row * (box_h + 10)
                
                # 박스 그리기
                pdf_canvas.rect(x, y_box, box_w, box_h, stroke=1, fill=0)
                
                # 페이지 번호
                pdf_canvas.setFont("Helvetica-Bold", 12)
                pdf_canvas.drawCentredText(x + box_w/2, y_box + box_h/2 + 5, f"Page {page_num}")
                
                # 위치
                pdf_canvas.setFont("Helvetica", 10)
                pdf_canvas.drawCentredText(x + box_w/2, y_box + box_h/2 - 8, f"R{row+1}C{col+1}")
                
                page_num += 1
        
        y_pos -= (self.pages_y * (box_h + 10)) + 30
        
        # 문제 해결
        pdf_canvas.setFont("Helvetica", 12)
        troubleshooting = [
            "🔧 문제 해결:",
            f"• 마커가 {self.markerLength*1000:.0f}mm가 아니면 → 프린터 설정 재확인",
            "• 페이지들이 안 맞으면 → 자르기 정확도 확인",
            "• 마커 인식 안 되면 → 조명 및 초점 확인",
        ]
        
        for item in troubleshooting:
            pdf_canvas.drawString(50, y_pos, item)
            y_pos -= 16
        
        pdf_canvas.save()
        print(f"✅ 조립 가이드 생성: {filename}")

class ARUCOBoardPose:
    def __init__(self) -> None:
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.size = (5, 7)
        self.markerLength = 0.060  # ✅ 60mm로 통일
        self.markerSeparation = 0.005  # ✅ 5mm로 통일
        self.board = cv2.aruco.GridBoard(self.size, self.markerLength, self.markerSeparation, self.dictionary, None)
        self.detectorParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detectorParams)

    def run(self, camera_k, camera_d, imgraw):
        corners, ids, rej = self.detector.detectMarkers(imgraw)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(imgraw, corners, ids)
            objPoints, imgPoints = self.board.matchImagePoints(corners, ids, None, None)
            retval, rvc, tvc = cv2.solvePnP(objPoints, imgPoints, camera_k, camera_d, None, None, False)
            R, _ = cv2.Rodrigues(rvc)
            if objPoints is not None:
                cv2.drawFrameAxes(imgraw, camera_k, camera_d, rvc, tvc, 0.1, 3)
            return tvc, R
        return None

if __name__ == "__main__":
    print("=== ArUco Board PDF Generator (정확한 크기) ===")
    generator = ARUCOBoardPDFGenerator()
    generator.generate_multi_page_pdf("aruco_board_5*5_60mm_accurate.pdf")
    
    print("\n✅ 완료!")
    print("📄 aruco_board_60mm_accurate.pdf - 인쇄할 페이지들")
    print("📋 aruco_board_60mm_accurate_assembly_guide.pdf - 조립 가이드")
    print("\n🔍 중요 검증 단계:")
    print("1. 인쇄 후 첫 번째 마커를 자로 측정")
    print("2. 정확히 60mm가 나오는지 확인")
    print("3. 안 맞으면 프린터 설정에서 '실제 크기' 재확인")