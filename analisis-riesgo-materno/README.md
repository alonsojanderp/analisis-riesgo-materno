# Análisis y Modelado Predictivo de Riesgo de Muerte Materna (Dataset Sintético)

Proyecto en Python que simula el flujo de trabajo de Machine Learning para un dataset sintético de salud con outcome **Muerte_Materna**.  
Se contrasta **inferencia (interpretación de coeficientes/OR)** vs **predicción (rendimiento fuera de muestra)**.

## Ejecutar

```bash
python analisis_riesgo_materno.py
```

## Resultados (semilla fija)

- **Prevalencia (Muerte_Materna=1)**: 0.0120
- **Odds Ratio (Riesgo_Previo)**: 1.959

### Robustez (Validación Cruzada en Train, K=5)
- **AUC promedio (CV 5-fold)**: 0.626 ± 0.133

### Rendimiento Final (Test Set 30%)
- **AUC (ROC)**: 0.691  
- **Sensibilidad (Recall)**: 0.727  
- **VPP (Precision)**: 0.025

---

## 4) Reflexión Epidemiológica (respuestas solicitadas)

### Inferencia: ¿Cuál es el Odds Ratio (OR) para Riesgo_Previo y cómo se interpreta clínicamente?
El **OR para `Riesgo_Previo` es 1.959**.  
Interpretación: ajustando por Edad, retraso en atención prenatal y presión arterial, tener **historial de alto riesgo** se asocia a un aumento de aproximadamente **2.0×** en las *odds* del outcome **Muerte_Materna=1**.  
Clínicamente, `Riesgo_Previo` actúa como un **marcador de vulnerabilidad previa**, útil para priorizar vigilancia y recursos al ingreso.

### Robustez: Compare el AUC Promedio de CV con el AUC Final en el Test Set
AUC CV (train) = **0.626** vs AUC test = **0.691**.

Si el **AUC de CV fuera mucho más bajo** que el AUC de test, sugeriría un problema en el **Bias-Variance Tradeoff**:
- Mayor probabilidad de **alta varianza/inestabilidad**: el rendimiento cambia mucho según cómo se dividan los datos.
- Posible “test set fácil” por azar (común en eventos raros) o incluso **data leakage** si el pipeline usara información que no estaría disponible al ingreso.

La implicancia práctica es que el rendimiento aparente no es confiable sin validación robusta.

### Prioridad Clínica: Minimizar Falsos Negativos
Si el hospital prioriza **no dejar pacientes de alto riesgo sin vigilancia**, la métrica crítica a maximizar es la **Sensibilidad (Recall)**.  
En este experimento, el valor obtenido fue: **Sensibilidad = 0.727**.

> En operación real, se ajusta el **umbral** (bajándolo) para subir sensibilidad, aceptando más falsos positivos, según capacidad de vigilancia.

### Conclusión Ejecutiva (Gerente del Hospital)
**Fortaleza:** el modelo entrega una probabilidad de riesgo basada en variables disponibles al ingreso, con un rendimiento global razonable (AUC) y una señal clínica interpretable (OR elevado para `Riesgo_Previo`).  
**Debilidad:** al ser un evento raro, la clasificación depende fuertemente del umbral; mejorar sensibilidad suele reducir VPP (más falsas alarmas). Antes de implementar, se recomienda fijar un umbral clínico, validar en datos reales y monitorear desempeño en el tiempo.
