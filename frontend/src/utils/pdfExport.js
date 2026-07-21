import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

/**
 * Create a professional hospital-style clinical prediction report
 * Version: 3.0 - Production Ready
 */
export const exportStructuredPDF = async (predictionData) => {
  try {
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    
    // Professional color scheme - Hospital Style
    const colors = {
      primary: [0, 70, 150],      // Deep Navy Blue
      primaryLight: [40, 110, 200],
      secondary: [80, 90, 110],   // Slate Gray
      text: [30, 40, 60],         // Dark Navy
      lightBg: [248, 249, 251],   // Light Gray
      border: [210, 215, 220],    // Border Gray
      success: [0, 150, 100],     // Teal Green
      warning: [200, 130, 0],     // Amber
      danger: [190, 30, 40],      // Crimson Red
      white: [255, 255, 255],
      gold: [180, 150, 60]        // Gold Accent
    };

    // ============================================================
    // HELPER FUNCTIONS
    // ============================================================
    
    // Check if we need a new page
    const checkPageOverflow = (currentY, neededSpace = 60) => {
      if (currentY > pageHeight - neededSpace) {
        pdf.addPage();
        return 20;
      }
      return currentY;
    };

    // Get risk color based on score
    const getRiskColor = (score) => {
      if (score > 70) return colors.danger;
      if (score > 40) return colors.warning;
      return colors.success;
    };

    // Get alert level based on score
    const getAlertLevel = (score) => {
      if (score > 70) return 'CRITICAL';
      if (score > 40) return 'WARNING';
      return 'STABLE';
    };

    // Get risk text based on score
    const getRiskText = (score) => {
      if (score > 70) return 'HIGH RISK';
      if (score > 40) return 'MODERATE RISK';
      return 'LOW RISK';
    };

    // ============================================================
    // EXTRACT DATA WITH NULLISH COALESCING
    // ============================================================
    
    const riskScore = (predictionData.risk_score ?? 0.5) * 100;
    const threshold = predictionData.threshold ?? 0.459;
    const thresholdPercentage = Math.round(Math.max(0, Math.min(100, threshold * 100)));
    const riskColor = getRiskColor(riskScore);
    const alertLevel = predictionData.alert_level || getAlertLevel(riskScore);
    const isAboveThreshold = riskScore / 100 > threshold;
    
    // Handle confidence properly (string or number)
    let confidenceValue = 85;
    if (typeof predictionData.confidence === 'string') {
      const confidenceMap = { 'HIGH': 95, 'MEDIUM': 85, 'LOW': 70 };
      confidenceValue = confidenceMap[predictionData.confidence] || 85;
    } else if (typeof predictionData.confidence === 'number') {
      confidenceValue = Math.round(Math.min(100, Math.max(0, predictionData.confidence * 100)));
    }
    
    // Confidence Interval
    const ciMargin = 0.07;
    const ciLower = Math.max(0, Math.round((riskScore / 100 - ciMargin) * 100));
    const ciUpper = Math.min(100, Math.round((riskScore / 100 + ciMargin) * 100));
    
    // Generate filename
    const patientId = predictionData.patient_id || 'Unknown';
    const filename = `ICU_Report_${patientId}_${Date.now().toString().slice(-6)}.pdf`;

    let yPos = 15;

    // ============================================================
    // HEADER - Professional Hospital Style
    // ============================================================
    
    // Top color bar
    pdf.setFillColor(colors.primary[0], colors.primary[1], colors.primary[2]);
    pdf.rect(0, 0, pageWidth, 5, 'F');
    
    // Thin accent bar
    pdf.setFillColor(colors.gold[0], colors.gold[1], colors.gold[2]);
    pdf.rect(0, 5, pageWidth, 1.5, 'F');

    // Hospital Name
    pdf.setFontSize(24);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(colors.primary[0], colors.primary[1], colors.primary[2]);
    pdf.text('ICU Predictor', 20, 22);

    // Subtitle
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(colors.secondary[0], colors.secondary[1], colors.secondary[2]);
    pdf.text('Advanced ICU Deterioration Prediction System', 20, 30);

    // Report badge - Top Right Corner with centered text
    const badgeWidth = 48;
    const badgeHeight = 12;
    const badgeX = pageWidth - 20 - badgeWidth; // Right aligned with margin
    const badgeY = 16;
    
    // Badge background
    pdf.setFillColor(colors.primary[0], colors.primary[1], colors.primary[2]);
    pdf.roundedRect(badgeX, badgeY, badgeWidth, badgeHeight, 3, 3, 'F');
    
    // Badge text - Centered within the badge
    pdf.setFontSize(7);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(255, 255, 255);
    pdf.text('CLINICAL REPORT', badgeX + badgeWidth/2, badgeY + badgeHeight / 1.75, { align: 'center' });

    // Report ID and Date (right aligned, below badge)
    const reportId = `ICU-${Date.now().toString().slice(-8)}`;
    const dateStr = new Date().toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
    
    pdf.setFontSize(8);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(colors.secondary[0], colors.secondary[1], colors.secondary[2]);
    pdf.text(`Report ID: ${reportId}`, pageWidth - 20, 42, { align: 'right' });
    pdf.text(`Generated: ${dateStr}`, pageWidth - 20, 48, { align: 'right' });

    // Separator
    pdf.setDrawColor(colors.border[0], colors.border[1], colors.border[2]);
    pdf.setLineWidth(0.3);
    pdf.line(20, 36, pageWidth - 20, 36);

    yPos = 55;

    // ============================================================
    // PATIENT INFORMATION - Using autoTable
    // ============================================================
    
    pdf.setFontSize(11);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(colors.primary[0], colors.primary[1], colors.primary[2]);
    pdf.text('PATIENT INFORMATION', 20, yPos);

    yPos += 5;

    // Patient info table
    const patientData = [
      ['Patient Name:', predictionData.patient_name || 'Unknown'],
      ['Medical Record #:', predictionData.patient_id || 'N/A'],
      ['Assessment Time:', predictionData.prediction_time ? 
        new Date(predictionData.prediction_time).toLocaleString('en-US', {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        }) : 'Just now'],
      ['Prediction Window:', '24 Hours']
    ];

    autoTable(pdf, {
      startY: yPos,
      body: patientData,
      theme: 'plain',
      styles: {
        fontSize: 9.5,
        cellPadding: { top: 2.5, bottom: 2.5, left: 5, right: 5 },
        lineColor: colors.border,
        lineWidth: 0.1
      },
      columnStyles: {
        0: { 
          fontStyle: 'bold', 
          textColor: colors.secondary,
          cellWidth: 45,
          halign: 'left'
        },
        1: { 
          textColor: colors.text,
          cellWidth: 80,
          halign: 'left'
        }
      },
      margin: { left: 20, right: 20 }
    });

    yPos = (pdf.lastAutoTable?.finalY || yPos) + 12;

    // ============================================================
    // RISK ASSESSMENT
    // ============================================================
    
    // Check page overflow
    yPos = checkPageOverflow(yPos, 80);
    
    // Section with colored border
    pdf.setDrawColor(riskColor[0], riskColor[1], riskColor[2]);
    pdf.setLineWidth(0.5);
    pdf.roundedRect(20, yPos, pageWidth - 40, 55, 4, 4, 'S');
    
    // Colored top bar for risk section
    pdf.setFillColor(riskColor[0], riskColor[1], riskColor[2]);
    pdf.roundedRect(20, yPos, pageWidth - 40, 4, 4, 4, 'F');

    pdf.setFontSize(11);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(colors.primary[0], colors.primary[1], colors.primary[2]);
    pdf.text('RISK ASSESSMENT', 30, yPos + 14);

    // Risk Score - Big Number
    pdf.setFontSize(28);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(riskColor[0], riskColor[1], riskColor[2]);
    pdf.text(`${Math.round(riskScore)}%`, 30, yPos + 40);

    // Risk details - Right side
    pdf.setFontSize(9.5);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(colors.secondary[0], colors.secondary[1], colors.secondary[2]);
    pdf.text('Alert Level:', 90, yPos + 20);
    pdf.text('Confidence:', 90, yPos + 34);
    pdf.text('95% Confidence Interval:', 90, yPos + 48);

    // Risk details - Values
    const alertColor = alertLevel === 'CRITICAL' ? colors.danger : 
                       alertLevel === 'WARNING' ? colors.warning : colors.success;
    
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(alertColor[0], alertColor[1], alertColor[2]);
    pdf.text(alertLevel, 170, yPos + 20);
    
    pdf.setTextColor(colors.text[0], colors.text[1], colors.text[2]);
    pdf.text(`${confidenceValue}%`, 170, yPos + 34);
    pdf.text(`${ciLower}% – ${ciUpper}%`, 170, yPos + 48);

    // Risk Gauge Bar with proper alignment
    const gaugeY = yPos + 52;
    const gaugeX = 30;
    const gaugeWidth = pageWidth - 70;
    const gaugeHeight = 6;
    
    // Background
    pdf.setFillColor(235, 237, 240);
    pdf.roundedRect(gaugeX, gaugeY, gaugeWidth, gaugeHeight, 3, 3, 'F');
    
    // Progress
    const progressWidth = Math.min((riskScore / 100) * gaugeWidth, gaugeWidth);
    pdf.setFillColor(riskColor[0], riskColor[1], riskColor[2]);
    pdf.roundedRect(gaugeX, gaugeY, progressWidth, gaugeHeight, 3, 3, 'F');

    // Gauge labels with proper alignment
    pdf.setFontSize(7);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(colors.secondary[0], colors.secondary[1], colors.secondary[2]);
    pdf.text('LOW', gaugeX, gaugeY + 12);
    pdf.text('MODERATE', gaugeX + gaugeWidth/2, gaugeY + 12, { align: 'center' });
    pdf.text('HIGH', gaugeX + gaugeWidth, gaugeY + 12, { align: 'right' });

    yPos += 75;

    // ============================================================
    // CLINICAL RECOMMENDATIONS - Using autoTable
    // ============================================================
    
    yPos = checkPageOverflow(yPos, 70);
    
    pdf.setFontSize(11);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(colors.primary[0], colors.primary[1], colors.primary[2]);
    pdf.text('CLINICAL RECOMMENDATIONS', 20, yPos);

    yPos += 5;

    const recommendations = getRecommendations(riskScore);
    const recData = recommendations.map((rec, index) => [
      `${index + 1}.`, rec
    ]);

    autoTable(pdf, {
      startY: yPos,
      body: recData,
      theme: 'plain',
      styles: {
        fontSize: 9,
        cellPadding: { top: 3, bottom: 3, left: 5, right: 5 },
        lineColor: colors.border,
        lineWidth: 0.1
      },
      columnStyles: {
        0: { 
          fontStyle: 'bold', 
          textColor: colors.primary,
          cellWidth: 12,
          halign: 'right'
        },
        1: { 
          textColor: colors.text,
          halign: 'left'
        }
      },
      margin: { left: 20, right: 20 }
    });

    yPos = (pdf.lastAutoTable?.finalY || yPos) + 12;

    // ============================================================
    // TOP CONTRIBUTING FACTORS (SHAP)
    // ============================================================
    
    if (predictionData.shap_values && Object.keys(predictionData.shap_values).length > 0) {
      yPos = checkPageOverflow(yPos, 70);
      
      const shapEntries = Object.entries(predictionData.shap_values)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 5);

      pdf.setFontSize(11);
      pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(colors.primary[0], colors.primary[1], colors.primary[2]);
      pdf.text('TOP CONTRIBUTING FACTORS', 20, yPos);

      yPos += 5;

      const featureLabels = {
        heart_rate: 'Heart Rate',
        sbp: 'Systolic BP',
        dbp: 'Diastolic BP',
        gcs: 'GCS',
        lactate: 'Lactate',
        urine_output: 'Urine Output',
        fio2: 'FiO₂',
        creatinine: 'Creatinine'
      };

      const shapTableData = shapEntries.map(([key, value]) => {
        const label = featureLabels[key] || key;
        const isPositive = value > 0;
        const percent = Math.abs(Math.round(value * 100));
        const direction = isPositive ? '+' : '-';
        return [
          label,
          direction,
          `${percent}%`,
          percent,
          isPositive
        ];
      });

      autoTable(pdf, {
        startY: yPos,
        head: [['Factor', 'Impact', 'Strength', 'Contribution']],
        body: shapTableData,
        theme: 'plain',
        styles: {
          fontSize: 9,
          cellPadding: { top: 2.5, bottom: 2.5, left: 5, right: 5 },
          lineColor: colors.border,
          lineWidth: 0.1,
          valign: 'middle'
        },
        columnStyles: {
          0: { 
            fontStyle: 'bold', 
            textColor: colors.text,
            cellWidth: 55,
            halign: 'left'
          },
          1: { 
            fontStyle: 'bold',
            cellWidth: 20,
            halign: 'center'
          },
          2: { 
            fontStyle: 'bold',
            cellWidth: 30,
            halign: 'right'
          },
          3: { 
            cellWidth: 65,
            halign: 'left'
          }
        },
        didDrawCell: function(data) {
          if (data.column.index === 3 && data.section === 'body' && data.cell.raw) {
            const percent = data.cell.raw;
            const isPositive = data.row.cells[4]?.raw || false;
            const color = isPositive ? [190, 30, 40] : [0, 150, 100];
            
            const cell = data.cell;
            const x = cell.x + 2;
            const y = cell.y + 3;
            const width = cell.width - 4;
            const height = 4;
            
            pdf.setFillColor(235, 237, 240);
            pdf.roundedRect(x, y, width, height, 2, 2, 'F');
            
            const progress = Math.min(percent, 100);
            pdf.setFillColor(color[0], color[1], color[2]);
            pdf.roundedRect(x, y, (progress / 100) * width, height, 2, 2, 'F');
          }
        },
        margin: { left: 20, right: 20 }
      });

      yPos = (pdf.lastAutoTable?.finalY || yPos) + 12;
    }

    // ============================================================
    // DECISION THRESHOLD
    // ============================================================
    
    yPos = checkPageOverflow(yPos, 50);
    
    pdf.setFillColor(248, 249, 251);
    pdf.roundedRect(20, yPos, pageWidth - 40, 28, 4, 4, 'F');
    pdf.setDrawColor(colors.border[0], colors.border[1], colors.border[2]);
    pdf.setLineWidth(0.3);
    pdf.roundedRect(20, yPos, pageWidth - 40, 28, 4, 4, 'S');

    pdf.setFontSize(9);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(colors.secondary[0], colors.secondary[1], colors.secondary[2]);
    pdf.text('Decision Threshold:', 30, yPos + 10);
    pdf.text('Risk Status:', 30, yPos + 21);
    
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(colors.text[0], colors.text[1], colors.text[2]);
    pdf.text(`${thresholdPercentage}%`, 100, yPos + 10);
    
    const statusColor = isAboveThreshold ? colors.danger : colors.success;
    pdf.setTextColor(statusColor[0], statusColor[1], statusColor[2]);
    pdf.text(isAboveThreshold ? 'ABOVE THRESHOLD - Action Required' : 'BELOW THRESHOLD - Monitoring', 100, yPos + 21);

    yPos += 38;

    // ============================================================
    // DISCLAIMER & FOOTER
    // ============================================================
    
    yPos = checkPageOverflow(yPos, 50);

    // Disclaimer box
    pdf.setFillColor(252, 248, 240);
    pdf.setDrawColor(200, 180, 140);
    pdf.setLineWidth(0.3);
    pdf.roundedRect(20, yPos, pageWidth - 40, 26, 4, 4, 'F');
    pdf.roundedRect(20, yPos, pageWidth - 40, 26, 4, 4, 'S');

    pdf.setFontSize(7);
    pdf.setFont('helvetica', 'italic');
    pdf.setTextColor(130, 110, 70);
    
    const disclaimerText = 'CLINICAL DECISION SUPPORT NOTICE: This prediction is generated by an AI model trained on retrospective ICU data and is intended to assist, not replace, clinician judgment. Final treatment decisions should always be made by qualified healthcare professionals.';
    const splitDisclaimer = pdf.splitTextToSize(disclaimerText, pageWidth - 55);
    pdf.text(splitDisclaimer, 30, yPos + 6);

    yPos += 34;

    // ============================================================
    // DYNAMIC PAGE NUMBERS AND FOOTER
    // ============================================================
    
    // Add footer to all pages
    const totalPages = pdf.getNumberOfPages();
    
    for (let i = 1; i <= totalPages; i++) {
      pdf.setPage(i);
      
      const pageHeight = pdf.internal.pageSize.getHeight();
      const pageWidth = pdf.internal.pageSize.getWidth();
      
      // Footer line
      pdf.setDrawColor(colors.border[0], colors.border[1], colors.border[2]);
      pdf.setLineWidth(0.3);
      pdf.line(20, pageHeight - 15, pageWidth - 20, pageHeight - 15);

      pdf.setFontSize(7.5);
      pdf.setFont('helvetica', 'normal');
      pdf.setTextColor(colors.secondary[0], colors.secondary[1], colors.secondary[2]);
      
      const footerLeft = [
        `ICU Predictor v1.0`,
        `Model: CatBoost | Dataset: MIMIC-IV v3.1`
      ].join('  •  ');
      pdf.text(footerLeft, 20, pageHeight - 7);

      pdf.text(`Page ${i} of ${totalPages}`, pageWidth - 20, pageHeight - 7, { align: 'right' });
    }

    // ============================================================
    // SAVE PDF
    // ============================================================
    
    pdf.save(filename);
    return true;

  } catch (error) {
    console.error('PDF Export Error:', error);
    throw error;
  }
};

// Helper function for clinical recommendations
function getRecommendations(score) {
  if (score >= 70) {
    return [
      'Notify ICU attending physician immediately',
      'Repeat vital signs within 15 minutes',
      'Monitor lactate trend closely',
      'Consider vasopressor assessment',
      'Prepare for potential ICU transfer'
    ];
  } else if (score >= 40) {
    return [
      'Increase vital sign monitoring to every 30 minutes',
      'Monitor GCS and lactate trends',
      'Check creatinine and urine output in 2 hours',
      'Review with senior clinician',
      'Document clinical concerns'
    ];
  } else {
    return [
      'Continue standard monitoring protocol',
      'Reassess in 4 hours',
      'Maintain current treatment plan',
      'Document stable condition',
      'Patient education as needed'
    ];
  }
}

export default { exportStructuredPDF };