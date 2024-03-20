<img src="_images/localbip_logo.png" alt="biplot_logo" style="width: 150px;"/> 

# Manual Técnico Software


## Localbiplot - Biblioteca de código abierto para el análisis de datos de alta dimensión utilizando biplots por localidades, como soporte a aplicaciones de agricultura de precisión



### 1. Descripción general del sistema de información desarrollado

La biblioteca de código abierto desarrollada en Python para el análisis de datos de alta dimensión utilizando biplots por localidades (Localbiplot), como soporte a aplicaciones de agricultura de precisión, consiste en un conjunto de módulos diseñados para realizar un análisis basado en una extensión del biplot mediante la descomposición de Valores Singulares (SVD) a nivel de localidades. Su objetivo es resaltar, analizar e identificar patrones en subgrupos específicos de datos no estacionarios, utilizando relaciones tanto lineales como no lineales. Esto permite visualizar las relaciones entre muestras y variables en los diferentes conjuntos de ejes generados. La herramienta consta de varios módulos que se encargan de lo siguiente: i) Normalizar y centralizar datos; ii) Obtener representaciones de baja dimensión lineales (SVD) y no lineales (t-SNE) a nivel de todo el conjunto de datos; iii) Realizar análisis de subconjuntos de datos de alta y baja dimensión (clusters), basado en una variable específica del conjunto. Luego, se calcula el SVD local en cada grupo para obtener las matrices biplot; iv) Aplicar transformaciones afines con los parámetros que mejor escalan, traducen y rotan cada uno de los múltiples conjuntos de ejes generados para cada subespacio de los datos agrupados.



### 2. Requerimientos técnicos a nivel de hardware y software para instalar y operar el software desarrollado



localbiplot necesita Python >= 3.8  y acceso a internet para descargar las librerias

2.1 Instalar desde el código fuente


```python
!pip install -U git+https://github.com/Jectrianama/python-gcpds.localbiplot.git --quiet

```

2.2 Añadir la libreria  en su código como :


```python
import gcpds.localbiplot as lb
```

### 3. Entorno y/o Lenguaje de desarrollo utilizado

La biblioteca local biplot esta desarrollado utilizando python 3.11. Ademas, el núcleo principal de este software se encuentra en librerias como matplotlib, y scipy, diseñadas  para manejar todo lo relacionado con graficas y manejo de estructuras de datos para asi amplíar las capacidades de la herramienta.




### 4. Descripción de los diferentes programas (módulos) que hacen parte del sistema de información y su interrelación (teniendo en cuenta que un programa de computador puede estar compuesto de uno o varios módulos que cumplen funciones específicas)





### 5. Motores de bases de datos sobre los que fue desarrollado el sistema (en caso de que este basado en archivos planos se debe realizar una descripción de la forma como operan).

### 6. Diccionario de datos (descripción de las tablas, campos, tipos y tamaño de campos que la conforman, identificando llaves primarias y secundarias si es del caso)

### 7. Diagrama entidad relación de las estructuras de datos o en su defecto estructura de funcionamiento e interacción del sistema de archivos

### 8. Principales casos de uso


```python

```
