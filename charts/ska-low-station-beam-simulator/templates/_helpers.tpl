{{/*
Expand the name of the chart.
*/}}
{{- define "ska-low-station-beam-simulator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 59 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 59 chars because some Kubernetes name fields are limited
to 63 this (by the DNS naming spec), and we want to suffix the station ID.
*/}}
{{- define "ska-low-station-beam-simulator.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 59 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 59 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 59 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ska-low-station-beam-simulator.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "ska-low-station-beam-simulator.labels" -}}
helm.sh/chart: {{ include "ska-low-station-beam-simulator.chart" . }}
{{ include "ska-low-station-beam-simulator.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "ska-low-station-beam-simulator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ska-low-station-beam-simulator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Create the name of the service account to use
*/}}
{{- define "ska-low-station-beam-simulator.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{ default (include "ska-low-station-beam-simulator.fullname" .) .Values.serviceAccount.name }}
{{- else -}}
{{ default "default" .Values.serviceAccount.name }}
{{- end -}}
{{- end -}}
