variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Deployment environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "vpc_id" {
  description = "VPC ID where resources will be deployed"
  type        = string
}

variable "public_subnets" {
  description = "List of public subnet IDs"
  type        = list(string)
}

variable "private_subnets" {
  description = "List of private subnet IDs"
  type        = list(string)
}

variable "certificate_arn" {
  description = "ARN of the ACM certificate for HTTPS"
  type        = string
}

variable "task_cpu" {
  description = "CPU units for ECS task"
  type        = string
  default     = "1024"
}

variable "task_memory" {
  description = "Memory for ECS task (MB)"
  type        = string
  default     = "8192"
}

variable "desired_count" {
  description = "Number of ECS tasks to run"
  type        = number
  default     = 2
}

variable "model_path" {
  description = "Path to the model in S3 or EFS"
  type        = string
  default     = "/opt/ml/model"
}

variable "secret_key" {
  description = "Secret key for application"
  type        = string
  sensitive   = true
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the ALB"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default     = {}
}
