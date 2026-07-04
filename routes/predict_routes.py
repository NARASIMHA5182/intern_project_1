from flask import Blueprint, render_template, request, jsonify, current_app, url_for, Response, send_file, flash, redirect
from flask_login import login_required, current_user
import pandas as pd
import os
import io
import csv
from utils.helpers import get_project_root
from prediction.engine import predict

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict_form():
    """Render the prediction input form or compute decision on POST."""
    if request.method == 'POST':
        # Read form parameters (which are in snake_case)
        input_data = request.form.to_dict()
        
        # Run prediction
        try:
            result = predict(input_data)
        except Exception as e:
            current_app.logger.exception('Prediction failed')
            flash(f"Prediction computation failed: {str(e)}", "danger")
            return redirect(url_for('predict.predict_form'))

        # Save to database
        from database.models import Prediction
        from database import db
        
        pred_record = Prediction(
            user_id=current_user.id if current_user.is_authenticated else None,
            input_data=input_data,
            prediction=result['prediction'],
            probability=result['probability'],
            explanation_image=result['shap_image']
        )
        db.session.add(pred_record)
        db.session.commit()

        # Generate explanations and suggestions for results page
        from prediction.engine import clean_input_types
        mapped_input = clean_input_types(input_data)
        
        from utils.explainers import explain_prediction, generate_suggestions
        from prediction.engine import load_assets
        
        model, cleaner, transformer = load_assets()
        df = pd.DataFrame([mapped_input])
        df_clean = cleaner.transform(df)
        X_trans = transformer.transform(df_clean)
        
        explanation = explain_prediction(model, X_trans, mapped_input, transformer.feature_columns)
        suggestions = generate_suggestions(explanation['negative'], mapped_input)
        
        return render_template('result.html', pred=pred_record, explanation=explanation, suggestions=suggestions)
        
    return render_template('predict.html')

@predict_bp.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """Accept JSON payload, run model prediction, store result, and return response."""
    if not request.is_json:
        return jsonify({'error': 'Invalid input, JSON expected'}), 400
    input_data = request.get_json()
    # Basic validation: ensure required fields are present
    required_fields = [
        'Age', 'Gender', 'Occupation', 'Employment Type', 'Education', 'Annual Income',
        'Monthly Income', 'Monthly Expenses', 'Credit Score', 'Loan Amount', 'Existing Loans',
        'Debt Ratio', 'Years of Employment', 'Marital Status', 'Residence Type', 'Dependents',
        'Bank Balance', 'Savings', 'Investment', 'Loan History', 'Credit History'
    ]
    missing = [f for f in required_fields if f not in input_data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    # Run prediction
    try:
        result = predict(input_data)
    except Exception as e:
        current_app.logger.exception('Prediction failed')
        return jsonify({'error': str(e)}), 500

    # Save prediction to DB
    from database.models import Prediction
    pred_record = Prediction(
        user_id=current_user.id,
        input_data=input_data,
        prediction=result['prediction'],
        probability=result['probability'],
        explanation_image=result['shap_image']
    )
    from database import db
    db.session.add(pred_record)
    db.session.commit()

    # Return response with SHAP image URL
    shap_url = url_for('static', filename=result['shap_image'])
    return jsonify({
        'prediction': result['prediction'],
        'probability': result['probability'],
        'shap_image_url': shap_url
    })


@predict_bp.route('/export/excel')
@login_required
def export_excel():
    """Export prediction history as an Excel (.xlsx) file."""
    from database.models import Prediction
    if current_user.is_admin:
        preds = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    else:
        preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()

    rows = []
    for p in preds:
        row = {
            'ID': p.id,
            'Timestamp': p.timestamp,
            'Prediction': p.prediction,
            'Probability': p.probability,
        }
        if isinstance(p.input_data, dict):
            row.update(p.input_data)
        rows.append(row)

    df = pd.DataFrame(rows)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='predictions.xlsx'
    )


@predict_bp.route('/export/pdf')
@login_required
def export_pdf():
    """Export prediction history as CSV (lightweight alternative to PDF)."""
    from database.models import Prediction
    if current_user.is_admin:
        preds = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    else:
        preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Timestamp', 'Prediction', 'Probability'])
    for p in preds:
        writer.writerow([p.id, p.timestamp, p.prediction, round(p.probability * 100, 2)])
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=predictions_report.csv'}
    )


@predict_bp.route('/batch', methods=['GET', 'POST'])
@login_required
def batch_predict():
    """Batch CSV upload — run predictions on multiple applicants at once."""
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or not file.filename.endswith('.csv'):
            flash('Please upload a valid CSV file.', 'warning')
            return redirect(url_for('predict.batch_predict'))
        try:
            df = pd.read_csv(file)
            results = []
            for _, row in df.iterrows():
                input_data = row.to_dict()
                try:
                    result = predict(input_data)
                    results.append({
                        **input_data,
                        'prediction': result['prediction'],
                        'probability': result['probability']
                    })
                except Exception:
                    results.append({**input_data, 'prediction': 'Error', 'probability': 0.0})

            out_df = pd.DataFrame(results)
            output = io.StringIO()
            out_df.to_csv(output, index=False)
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=batch_results.csv'}
            )
        except Exception as e:
            current_app.logger.exception('Batch prediction failed')
            flash(f'Batch processing failed: {str(e)}', 'danger')
            return redirect(url_for('predict.batch_predict'))

    return render_template('batch.html')


@predict_bp.route('/batch/template')
@login_required
def download_template():
    """Download a blank CSV template for batch predictions."""
    headers = [
        'Age', 'Gender', 'Occupation', 'Employment Type', 'Education', 'Annual Income',
        'Monthly Income', 'Monthly Expenses', 'Credit Score', 'Loan Amount', 'Existing Loans',
        'Debt Ratio', 'Years of Employment', 'Marital Status', 'Residence Type', 'Dependents',
        'Bank Balance', 'Savings', 'Investment', 'Loan History', 'Credit History'
    ]
    output = io.StringIO()
    csv.writer(output).writerow(headers)
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=batch_template.csv'}
    )
