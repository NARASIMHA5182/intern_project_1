import io
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from utils.logger import logger

def export_predictions_to_excel(predictions_list):
    """
    Exports a list of prediction records to a styled Excel spreadsheet bytes stream.
    """
    logger.info("Exporting prediction records to Excel...")
    
    # Map predictions to a dictionary list
    data = []
    for pred in predictions_list:
        data.append({
            'ID': pred.id,
            'Timestamp': pred.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'User': pred.user.username if pred.user else 'Anonymous/Batch',
            'Age': pred.age,
            'Gender': pred.gender,
            'Occupation': pred.occupation,
            'Employment Type': pred.employment_type,
            'Education': pred.education,
            'Annual Income ($)': pred.annual_income,
            'Monthly Income ($)': pred.monthly_income,
            'Monthly Expenses ($)': pred.monthly_expenses,
            'Credit Score': pred.credit_score,
            'Loan Amount ($)': pred.loan_amount,
            'Existing Loans': pred.existing_loans,
            'Debt Ratio': pred.debt_ratio,
            'Years of Employment': pred.years_of_employment,
            'Marital Status': pred.marital_status,
            'Residence Type': pred.residence_type,
            'Dependents': pred.dependents,
            'Bank Balance ($)': pred.bank_balance,
            'Savings ($)': pred.savings,
            'Investment ($)': pred.investment,
            'Loan History': pred.loan_history,
            'Credit History': pred.credit_history,
            'Approval Status': pred.approval_status,
            'Probability': round(pred.probability, 4),
            'Confidence Score': round(pred.confidence_score, 4),
            'Risk Category': pred.risk_category
        })
        
    df = pd.DataFrame(data)
    
    output = io.BytesIO()
    # Write Excel sheet using openpyxl
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions_Audit')
        
        # Style sheet (auto-adjust column widths)
        workbook = writer.book
        worksheet = writer.sheets['Predictions_Audit']
        
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 10)
            
    output.seek(0)
    return output.getvalue()

def export_predictions_to_pdf(predictions_list):
    """
    Exports prediction records to a professional PDF document.
    """
    logger.info("Exporting prediction records to PDF...")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#002b66'), # Dark Blue
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#555555'),
        spaceAfter=20
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )
    
    table_body_style = ParagraphStyle(
        'TableBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#333333')
    )
    
    status_approved_style = ParagraphStyle(
        'Approved',
        parent=table_body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#198754') # Green
    )
    
    status_rejected_style = ParagraphStyle(
        'Rejected',
        parent=table_body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#dc3545') # Red
    )
    
    # 1. Document Header
    story.append(Paragraph("VANGUARD CREDIT SERVICES", title_style))
    story.append(Paragraph(
        f"Credit Card Approval System Audit Report | Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
        subtitle_style
    ))
    story.append(Spacer(1, 10))
    
    # 2. Table Data construction (Limit columns to fit letter page landscape/portrait width)
    # We display key columns: Date, Applicant, Score, Income, Debt, Status, Prob
    headers = ["Date", "User", "Credit Score", "Income ($)", "Debt Ratio", "Status", "Confidence"]
    
    table_data = [[Paragraph(h, table_header_style) for h in headers]]
    
    for pred in predictions_list[:100]: # Cap PDF visual report to top 100 entries
        user_name = pred.user.username if pred.user else 'Batch/Anon'
        date_str = pred.created_at.strftime('%Y-%m-%d')
        
        status_style = status_approved_style if pred.approval_status == 'Approved' else status_rejected_style
        
        row = [
            Paragraph(date_str, table_body_style),
            Paragraph(user_name, table_body_style),
            Paragraph(str(pred.credit_score), table_body_style),
            Paragraph(f"{pred.annual_income:,.0f}", table_body_style),
            Paragraph(f"{pred.debt_ratio:.2f}", table_body_style),
            Paragraph(pred.approval_status, status_style),
            Paragraph(f"{pred.probability * 100:.1f}%", table_body_style),
        ]
        table_data.append(row)
        
    # Build Table
    # Letter width is 612. Margins are 36 each side, leaving 540 pt width.
    col_widths = [70, 85, 75, 85, 75, 75, 75]
    
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#002b66')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 8),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ('TOPPADDING', (0,1), (-1,-1), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e9ecef')),
    ]))
    
    story.append(t)
    
    # 3. Build document
    doc.build(story)
    
    buffer.seek(0)
    return buffer.getvalue()
