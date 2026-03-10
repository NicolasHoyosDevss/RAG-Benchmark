# Plan de Implementacion: Metricas de Uso Multiproveedor (sin acople a OpenAI)

Fecha: 2026-03-10
Contexto: Migrar la captura de `execution_time`, `input_tokens`, `output_tokens`, `total_cost` para que funcione con cualquier proveedor soportado por LangChain (OpenAI, Anthropic, Gemini, endpoints OpenAI-compatible, etc.) sin depender de `get_openai_callback`.

## 1. Confirmacion tecnica (Context7)

Esta confirmacion se hizo con Context7 (disponible en esta sesion):

1. `AIMessage.usage_metadata` es el formato estandar de LangChain para tokens (cross-provider, cuando el proveedor lo entrega).
2. `UsageMetadataCallbackHandler` y `get_usage_metadata_callback()` son los mecanismos recomendados para agregar uso de tokens entre invocaciones y modelos.
3. LangChain advierte que con `ChatOpenAI` y `model_provider="openai"` contra proxies/endpoints OpenAI-compatible, campos no estandar del proveedor pueden no preservarse.
4. `get_openai_callback` / `OpenAICallbackHandler` pertenece a `langchain_community.callbacks.openai_info`, orientado a OpenAI (incluye tabla interna de precios por modelo OpenAI/Azure).

Referencias consultadas via Context7:

- https://docs.langchain.com/oss/python/langchain/models (Token usage)
- https://docs.langchain.com/oss/python/langchain/messages (AIMessage usage_metadata)
- https://docs.langchain.com/oss/python/integrations/chat/openai (warning de compatibilidad)
- https://reference.langchain.com/python/langchain_core/callbacks/#langchain_core.callbacks.usage.UsageMetadataCallbackHandler
- https://reference.langchain.com/python/langchain_community/callbacks/#langchain_community.callbacks.manager.get_openai_callback

## 2. Respuesta a la duda clave

Pregunta: si el dato viene de "OpenAI API response", eso significa solo OpenAI o tambien APIs OpenAI-compatible?

Respuesta:

1. Tokens: puede funcionar con APIs OpenAI-compatible solo si exponen `usage` en formato esperado y LangChain lo mapea a `usage_metadata`.
2. Costo con `get_openai_callback`: esta acoplado a OpenAI/Azure por tabla de precios interna. No es una solucion universal para cualquier proveedor.
3. Conclusion: para un enfoque multiproveedor real, usar `usage_metadata` (core) + capa de pricing propia.

## 3. Estado actual del repo

Acoples detectados a `get_openai_callback`:

- `src/rag/simple.py`
- `src/rag/hybrid.py`
- `src/rag/hyde.py`
- `src/rag/rewriter.py`
- `src/rag/pageindex.py`

Consumo de metadata de performance (ya agnostico si el contrato se mantiene):

- `src/evaluation/ragas_evaluator.py`

## 4. Objetivo de la migracion

Mantener el contrato actual hacia evaluacion (`metadata.execution_time`, `input_tokens`, `output_tokens`, `total_cost`) pero con captura de uso compatible con multiples proveedores.

## 5. Diseno propuesto

### 5.1 Capa comun de metrica de uso

Crear modulo nuevo:

- `src/common/usage_metrics.py`

Responsabilidades:

1. Extraer tokens desde `AIMessage.usage_metadata` (ruta primaria).
2. Fallback opcional a `response_metadata` cuando aplique.
3. Retornar estructura unificada:
   - `input_tokens: int`
   - `output_tokens: int`
   - `total_tokens: int`
   - `usage_source: str` (ej: `usage_metadata`, `response_metadata`, `missing`)

API sugerida:

```python
from typing import Any, Dict


def extract_usage_from_ai_message(message: Any) -> Dict[str, int | str]:
    ...
```

### 5.2 Capa comun de pricing multiproveedor

Crear modulo nuevo:

- `src/common/pricing.py`

Responsabilidades:

1. Calcular costo a partir de tokens + tabla de precios configurable.
2. Soportar clave por proveedor y modelo (`provider:model`).
3. Tolerar modelos sin precio configurado (retornar `None`/`0.0` + bandera).

API sugerida:

```python
from typing import Optional


def estimate_cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    ...
```

Configuracion sugerida (una opcion):

- `config/model_pricing.json`

Ejemplo:

```json
{
  "openai:gpt-5": {"input_per_1k": 0.00125, "output_per_1k": 0.01},
  "anthropic:claude-sonnet-4-6": {"input_per_1k": 0.003, "output_per_1k": 0.015}
}
```

### 5.3 Provider/model identity

Extender metadata de modelos en `src/common/model_provider.py` (o helper asociado) para exponer:

- `provider`
- `model_id`
- `model_name`

Esto permite calcular costo sin suposiciones OpenAI-only.

## 6. Plan de cambios por archivo

### Fase A - Infraestructura comun

1. Agregar `src/common/usage_metrics.py`.
2. Agregar `src/common/pricing.py`.
3. Agregar `config/model_pricing.json`.

### Fase B - Migrar RAGs

En cada RAG (`simple`, `hybrid`, `hyde`, `rewriter`, `pageindex`):

1. Quitar `get_openai_callback`.
2. Invocar LLM de forma normal (`response = llm.invoke(...)`).
3. Extraer tokens usando `extract_usage_from_ai_message(response)`.
4. Calcular costo con `estimate_cost_usd(...)`.
5. Mantener contrato de salida actual:
   - `metadata.input_tokens`
   - `metadata.output_tokens`
   - `metadata.total_cost`
   - `metadata.execution_time`
6. Agregar campos utiles opcionales:
   - `metadata.usage_source`
   - `metadata.cost_estimation_source` (`configured_pricing` / `missing_pricing`)

### Fase C - Compatibilidad evaluador

1. No romper `src/evaluation/ragas_evaluator.py`.
2. Verificar que agregados (`overall_stats`) sigan funcionando con costos faltantes (`0.0` o `None` normalizado).

### Fase D - Validacion

1. Smoke test de cada RAG con un modelo OpenAI.
2. Smoke test con al menos un proveedor no OpenAI (segun entorno disponible).
3. Confirmar que `results/*.json` incluye tiempos, tokens y costo en ambos escenarios.

## 7. Reglas de fallback

1. Si no hay `usage_metadata`:
   - tokens en `0`
   - `usage_source="missing"`
2. Si no hay precio configurado:
   - `total_cost=0.0` (o `None` si prefieres distinguir explicitamente)
   - `cost_estimation_source="missing_pricing"`
3. Nunca fallar la ejecucion por ausencia de usage o pricing.

## 8. Criterios de aceptacion

1. El codigo no depende de `get_openai_callback` en los modulos RAG.
2. Tokens salen de `AIMessage.usage_metadata` cuando exista.
3. Pipeline de evaluacion no se rompe y conserva columnas/llaves esperadas.
4. Costo no depende de tabla interna OpenAI de LangChain; se usa capa configurable propia.
5. Con proveedor no OpenAI, se registra uso y costo (si hay pricing configurado) o fallback claro (si no lo hay).

## 9. Riesgos y notas

1. Algunos proveedores no entregan usage en todos los modos (ej. streaming sin opt-in).
2. Proxies OpenAI-compatible pueden omitir campos o usar formatos extendidos no mapeados por `ChatOpenAI`.
3. Para proveedores con integracion nativa LangChain (Anthropic, Gemini, OpenRouter, LiteLLM), preferir wrapper especifico cuando sea posible.

## 10. Checklist operativo para el otro chat

1. Implementar `usage_metrics.py`.
2. Implementar `pricing.py`.
3. Crear `config/model_pricing.json` inicial.
4. Migrar 5 modulos RAG.
5. Ejecutar validaciones minimas y revisar JSON de resultados.
6. Ajustar pricing segun modelos reales del benchmark.

---

Notas de implementacion rapida (snippet orientativo):

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()
response = model.invoke(messages, config={"callbacks": [callback]})

usage = response.usage_metadata or {}
input_tokens = int(usage.get("input_tokens", 0) or 0)
output_tokens = int(usage.get("output_tokens", 0) or 0)
```

```python
# Alternativa con context manager
from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as cb:
    response = model.invoke(messages)
# cb.usage_metadata agregado por modelo
```
