# CreditGuard_Modeling_Adventures

# CreditGuard: Desbravando as Fronteiras da Avaliação de Crédito#

# Importar bibliotecas necessárias

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Separar 70% da base para treinamento e 30% para validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Definir um objeto com a função da árvore de decisão e treinar o modelo com os dados de treinamento
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_train, y_train)

# 3. Visualizar a árvore de decisão
plt.figure(figsize=(15, 10))
plot_tree(modelo_arvore, filled=True, feature_names=X.columns, class_names=['aprovado', 'reprovado'])
plt.show()

# 4. Produzir uma visualização da matriz de confusão para a base de treinamento
y_pred_train = modelo_arvore.predict(X_train)
matriz_confusao_train = confusion_matrix(y_train, y_pred_train)

# Criar visualização da matriz de confusão para a base de treinamento
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_train, annot=True, fmt='d', cmap='Blues', xticklabels=['aprovado', 'reprovado'],
            yticklabels=['aprovado', 'reprovado'])
plt.title('Matriz de Confusão - Base de Treinamento')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# Calcular a acurácia na base de treinamento
acuracia_treino = accuracy_score(y_train, y_pred_train)
print(f'Acurácia na base de treinamento: {acuracia_treino:.4f}')

# 5. Classificar a base de teste de acordo com a árvore treinada
y_pred_test = modelo_arvore.predict(X_val)

# 6. Produzir uma visualização da matriz de confusão para a base de teste
matriz_confusao_teste = confusion_matrix(y_val, y_pred_test)

# Criar visualização da matriz de confusão para a base de teste
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_teste, annot=True, fmt='d', cmap='Blues', xticklabels=['aprovado', 'reprovado'],
            yticklabels=['aprovado', 'reprovado'])
plt.title('Matriz de Confusão - Base de Teste')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# Calcular a acurácia na base de teste
acuracia_teste = accuracy_score(y_val, y_pred_test)
print(f'Acurácia na base de teste: {acuracia_teste:.4f}')

# 7. Treinar uma nova árvore com número mínimo de observações por folha de 5 e máximo de profundidade de 10
modelo_arvore_ajustada = DecisionTreeClassifier(min_samples_leaf=5, max_depth=10, random_state=123)
modelo_arvore_ajustada.fit(X_train, y_train)

# 8. Avaliar a matriz de confusão da nova árvore
y_pred_ajustada = modelo_arvore_ajustada.predict(X_val)
matriz_confusao_ajustada = confusion_matrix(y_val, y_pred_ajustada)

# Criar visualização da matriz de confusão para a árvore ajustada
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_ajustada, annot=True, fmt='d', cmap='Blues', xticklabels=['aprovado', 'reprovado'],
            yticklabels=['aprovado', 'reprovado'])
plt.title('Matriz de Confusão - Árvore Ajustada')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# 9. Observar a distribuição da predição
proporcao_maus = np.mean(y_pred_ajustada == 1)
print(f'Proporção de proponentes classificados como "maus": {proporcao_maus:.4f}')

# 10. Calcular a acurácia se todos os contratos fossem classificados como "bons"
acuracia_todos_bons = accuracy_score(y_val, np.zeros_like(y_val))
print(f'Acurácia se todos os contratos fossem classificados como "bons": {acuracia_todos_bons:.4f}')
```
